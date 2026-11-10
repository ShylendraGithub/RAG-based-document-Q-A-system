"""Hybrid retrieval: BM25 keyword search + vector search, merged with Reciprocal Rank Fusion.

Why both?  Keyword search is exact (error codes, names, "Section 4.2") but blind to paraphrase.
Vector search handles paraphrase but can miss exact tokens. RRF merges the two *rankings*, so
BM25 scores and cosine similarities (which live on different scales) never have to be compared:

    score(chunk) = sum over rankings of 1 / (rrf_k + rank)
"""

from __future__ import annotations

from dataclasses import dataclass

from .embedders import Embedder
from .store import Store, StoredChunk

MODES = ("hybrid", "keyword", "vector")


@dataclass
class Hit:
    chunk: StoredChunk
    score: float  # RRF score (hybrid), 1/(k+rank) (single-list modes)
    cosine: float  # vector similarity to the query, always computed (used for the answerability gate)
    keyword_rank: int | None  # 1-based rank in the BM25 list, None if absent
    vector_rank: int | None


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, *, candidates: int = 40, rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(
        self, query: str, k: int = 5, mode: str = "hybrid", sources: list[str] | None = None
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        doc_ids: list[int] | None = None
        if sources:
            doc_ids = self.store.resolve_sources(sources)
            if not doc_ids:
                return []
        if self.store.count_chunks() == 0 or not query.strip():
            return []

        qvec = self.embedder.embed([query], kind="query")[0]
        vec_ranking: list[int] = []
        kw_ranking: list[int] = []
        if mode in ("hybrid", "vector"):
            vec_ranking = [cid for cid, _ in self.store.vector_search(qvec, self.candidates, doc_ids)]
        if mode in ("hybrid", "keyword"):
            kw_ranking = self.store.fts_search(query, self.candidates, doc_ids)

        if mode == "hybrid":
            fused = rrf([kw_ranking, vec_ranking], self.rrf_k)
            ordered = sorted(fused, key=lambda c: (-fused[c], c))
            scores = fused
        else:
            ordered = kw_ranking if mode == "keyword" else vec_ranking
            scores = {cid: 1.0 / (self.rrf_k + r) for r, cid in enumerate(ordered, start=1)}

        top = ordered[:k]
        chunks = self.store.get_chunks(top)
        cosines = self.store.cosines(top, qvec)
        kw_pos = {cid: r for r, cid in enumerate(kw_ranking, start=1)}
        vec_pos = {cid: r for r, cid in enumerate(vec_ranking, start=1)}
        return [
            Hit(chunks[cid], scores[cid], cosines.get(cid, 0.0), kw_pos.get(cid), vec_pos.get(cid))
            for cid in top
            if cid in chunks
        ]
