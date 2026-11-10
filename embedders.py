"""Embedding back-ends behind one tiny interface.

    embedder.name      stable id, stored in the database so a store built with one model can't be
                       queried with another (vectors from different models are not comparable)
    embedder.embed(texts, kind="document"|"query") -> float32 array (n, dim), L2-normalised
    embedder.default_min_score   cosine below which a passage is treated as unrelated, or None

Bundled:
  * HashingEmbedder: offline, deterministic, numpy only. It is **lexical** (stemmed word and
    bigram feature hashing), not semantic: it will not know that "holiday" ~ "vacation". It exists
    so the whole system runs and is testable with no model download or API key.
  * OpenAICompatibleEmbedder: any ``/v1/embeddings`` endpoint. Real semantic vectors.
    Verified against a local stub server only, never against a live service.
  * SentenceTransformerEmbedder: local model via sentence-transformers. Written but never run here
    (the library wasn't installed in the build environment).
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
import zlib
from collections import Counter
from typing import Protocol

import numpy as np

from .text import content_terms


class EmbeddingError(Exception):
    """An embedding request failed."""


class Embedder(Protocol):
    name: str
    default_min_score: float | None

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    default_min_score: float | None = 0.12

    def __init__(self, dim: int = 2048):
        if dim < 64:
            raise ValueError("dim must be at least 64")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            terms = content_terms(text)
            feats: Counter[str] = Counter(terms)
            bigrams: Counter[str] = Counter(f"{a}_{b}" for a, b in zip(terms, terms[1:]))
            for weight, counter in ((1.0, feats), (0.5, bigrams)):
                for feat, count in counter.items():
                    raw = feat.encode("utf-8")
                    idx = zlib.crc32(raw) % self.dim
                    sign = 1.0 if zlib.crc32(raw, 0x9E3779B9) & 1 else -1.0
                    out[row, idx] += sign * weight * (1.0 + math.log(count))
        return _normalize(out)


class OpenAICompatibleEmbedder:
    default_min_score: float | None = None  # thresholds are model-specific: tune with the eval harness

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        self.model = model
        self.name = f"openai:{model}"
        self.api_key = api_key or os.environ.get("EMBEDDINGS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EmbeddingError("Set EMBEDDINGS_API_KEY (or OPENAI_API_KEY) to use remote embeddings")
        self.base_url = (base_url or os.environ.get("EMBEDDINGS_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def _post(self, inputs: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": inputs}).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise EmbeddingError(f"Embeddings API error {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"Embeddings request failed: {exc}") from exc
        try:
            rows = sorted(payload["data"], key=lambda d: d["index"])
            vectors = [r["embedding"] for r in rows]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("Unexpected response shape from the embeddings API") from exc
        if len(vectors) != len(inputs):
            raise EmbeddingError(f"Asked for {len(inputs)} embeddings, received {len(vectors)}")
        return vectors

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        prefix = self.query_prefix if kind == "query" else self.document_prefix
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._post([prefix + t for t in texts[i : i + self.batch_size]]))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize(np.asarray(vectors, dtype=np.float32))


class SentenceTransformerEmbedder:  # pragma: no cover  (library not installed where this was written)
    default_min_score: float | None = None

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("pip install sentence-transformers to use local models") from exc
        self.model_name = model
        self.name = f"st:{model}"
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        return _normalize(np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32))


def make_embedder(spec: str = "hashing") -> Embedder:
    """``hashing`` | ``hashing:4096`` | ``openai:<model>`` | ``st:<model>``

    Also accepts an embedder's own ``.name`` (e.g. ``hashing-2048``), which is what an existing
    index stores, so tools can reopen an index without being told how it was built."""
    kind, _, arg = spec.partition(":")
    if kind.startswith("hashing"):
        suffix = kind[len("hashing"):].lstrip("-") or arg
        if not suffix.isdigit() and suffix != "":
            raise ValueError(f"Bad hashing embedder spec {spec!r}")
        return HashingEmbedder(int(suffix) if suffix else 2048)
    if kind == "openai":
        return OpenAICompatibleEmbedder(arg or "text-embedding-3-small")
    if kind == "st":
        return SentenceTransformerEmbedder(arg or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedder {spec!r} (use hashing, openai:<model> or st:<model>)")
