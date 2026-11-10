import threading
from pathlib import Path

import numpy as np
import pytest

from rag_docqa.embedders import HashingEmbedder
from rag_docqa.ingest import ingest_path
from rag_docqa.retriever import Retriever
from rag_docqa.store import Store

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


class ScriptedLLM:
    """Returns pre-written replies in order and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages):
        self.calls.append((system, [dict(m) for m in messages]))
        return self.replies.pop(0) if self.replies else "I don't know based on the provided documents."


class FakeEmbedder:
    """Returns one fixed query vector; lets tests control vector ranking exactly."""

    name = "fake-3"
    default_min_score = None

    def __init__(self, query_vec=(0.0, 1.0, 0.0)):
        self.query_vec = np.asarray(query_vec, dtype=np.float32)

    def embed(self, texts, kind="document"):
        return np.tile(self.query_vec, (len(texts), 1))


def make_pdf(pages: list[str]) -> bytes:
    """A minimal but valid multi-page PDF (Helvetica text), so loaders are tested on real PDF bytes."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = "BT /F1 12 Tf 72 720 Td 14 TL " + " T* ".join(f"({line}) Tj" for line in esc.split("\n")) + " ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream".encode())
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@pytest.fixture
def emb():
    return HashingEmbedder()


@pytest.fixture
def store(emb):
    s = Store(":memory:", emb.name)
    yield s
    s.close()


@pytest.fixture
def indexed(store, emb):
    """A store with the bundled sample documents ingested."""
    results = ingest_path(SAMPLE_DOCS, store, emb)
    assert all(r.status == "added" for r in results), results
    return store


@pytest.fixture
def retriever(indexed, emb):
    return Retriever(indexed, emb)


def run_in_thread(httpd):
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return t
