import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from rag_docqa.chunker import Chunk
from rag_docqa.embedders import EmbeddingError, HashingEmbedder, OpenAICompatibleEmbedder, make_embedder
from rag_docqa.retriever import MODES, Retriever, rrf
from rag_docqa.store import Store, StoreError

from conftest import FakeEmbedder


def vecs(*rows):
    return np.asarray(rows, dtype=np.float32)


def add(store, source, texts, vectors, title="T", sha="h"):
    chunks = [Chunk(i, t, section=f"S{i}") for i, t in enumerate(texts)]
    return store.add_document(source, title, sha, chunks, vectors)


# ---- store -----------------------------------------------------------------------------------

def test_add_and_read_back(store):
    add(store, "a.md", ["alpha text", "beta text"], vecs([1, 0, 0], [0, 1, 0]), title="Doc A")
    docs = store.documents()
    assert [(d.source, d.title, d.n_chunks) for d in docs] == [("a.md", "Doc A", 2)]
    assert store.count_chunks() == 2 and store.get_hash("a.md") == "h" and store.get_hash("zzz") is None
    ids = store.fts_search("alpha", 5)
    chunk = store.get_chunks(ids)[ids[0]]
    assert chunk.source == "a.md" and chunk.section == "S0" and chunk.text == "alpha text"


def test_replace_removes_old_chunks_from_every_index(store):
    add(store, "a.md", ["zebra stripes"], vecs([1, 0, 0]))
    add(store, "a.md", ["giraffe neck"], vecs([0, 1, 0]), sha="h2")
    assert store.count_chunks() == 1 and store.get_hash("a.md") == "h2"
    assert store.fts_search("zebra", 5) == []  # keyword index cleaned
    assert store.fts_search("giraffe", 5) != []
    (top,) = store.vector_search(np.asarray([1, 0, 0], dtype=np.float32), 5)
    assert top[1] == pytest.approx(0.0)  # old vector is gone: only the new one (orthogonal) remains


def test_delete_cascades_everywhere(store):
    add(store, "a.md", ["zebra"], vecs([1, 0, 0]))
    add(store, "b.md", ["giraffe"], vecs([0, 1, 0]))
    assert store.delete_document("a.md") is True
    assert store.delete_document("a.md") is False
    assert [d.source for d in store.documents()] == ["b.md"]
    assert store.fts_search("zebra", 5) == []
    assert len(store.vector_search(np.asarray([1, 0, 0], dtype=np.float32), 5)) == 1
    assert store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1


def test_failed_replace_rolls_back_and_keeps_the_old_version(store):
    add(store, "a.md", ["original text"], vecs([1, 0, 0]))
    bad = [Chunk(0, "new text"), Chunk(1, None)]  # NOT NULL violation on the second chunk
    with pytest.raises(sqlite3.IntegrityError):
        store.add_document("a.md", "T", "h2", bad, vecs([0, 1, 0], [0, 0, 1]))
    assert store.get_hash("a.md") == "h"
    assert store.count_chunks() == 1
    assert store.fts_search("original", 5) != [] and store.fts_search("new", 5) == []
    assert len(store.vector_search(np.asarray([1, 0, 0], dtype=np.float32), 5)) == 1


def test_input_validation(store):
    with pytest.raises(StoreError):
        store.add_document("a", "T", "h", [Chunk(0, "x")], vecs([1, 0], [0, 1]))
    with pytest.raises(StoreError):
        store.add_document("a", "T", "h", [], vecs())
    add(store, "a.md", ["x"], vecs([1, 0, 0]))
    with pytest.raises(StoreError, match="does not match"):
        add(store, "b.md", ["y"], vecs([1, 0]))
    with pytest.raises(StoreError, match="dims"):
        store.vector_search(np.asarray([1, 0], dtype=np.float32), 3)


def test_vector_search_order_filter_and_cache_invalidation(store):
    add(store, "a.md", ["a1", "a2"], vecs([1, 0, 0], [0.8, 0.6, 0]))
    add(store, "b.md", ["b1"], vecs([0, 1, 0]))
    q = np.asarray([1, 0, 0], dtype=np.float32)
    ranked = store.vector_search(q, 10)
    assert [round(s, 2) for _, s in ranked] == [1.0, 0.8, 0.0]
    only_b = store.vector_search(q, 10, doc_ids=store.resolve_sources(["b.md"]))
    assert len(only_b) == 1 and store.get_chunks([only_b[0][0]])[only_b[0][0]].source == "b.md"
    assert len(store.vector_search(q, 1)) == 1
    add(store, "c.md", ["c1"], vecs([1, 0, 0]))  # cache must see the new document
    assert len(store.vector_search(q, 10)) == 4


def test_empty_store_searches_return_nothing(store):
    assert store.vector_search(np.zeros(3, dtype=np.float32), 5) == []
    assert store.fts_search("anything", 5) == []
    assert store.documents() == [] and store.get_chunks([]) == {}


def test_fts_filter_by_document(store):
    add(store, "a.md", ["shared word here"], vecs([1, 0, 0]))
    add(store, "b.md", ["shared word there"], vecs([0, 1, 0]))
    assert len(store.fts_search("shared", 10)) == 2
    only_a = store.fts_search("shared", 10, doc_ids=store.resolve_sources(["a.md"]))
    assert len(only_a) == 1


def test_fts_uses_stemming_and_title_section_context(store):
    chunk = Chunk(0, "Approved claims are paid monthly.", section="Expense Rules")
    store.add_document("e.md", "Finance Handbook", "h", [chunk], vecs([1, 0, 0]))
    assert store.fts_search("approving claim", 5) != []  # porter: approving/approved, claim/claims
    assert store.fts_search("finance", 5) != []  # title is indexed even though the body never says it
    assert store.fts_search("expense", 5) != []  # so is the section


def test_persistence_and_embedder_binding(tmp_path):
    path = tmp_path / "idx.db"
    s = Store(path, "emb-a")
    add(s, "a.md", ["persisted"], vecs([1, 0, 0]))
    s.close()
    again = Store(path)  # no embedder given: reads the stored name
    assert again.embedder_name == "emb-a" and again.count_chunks() == 1
    again.close()
    with pytest.raises(StoreError, match="emb-a"):
        Store(path, "emb-b")
    Store(path, "emb-a").close()


def test_concurrent_reads_and_writes_do_not_corrupt(store):
    add(store, "seed.md", ["seed"], vecs([1, 0, 0]))
    errors = []

    def writer(n):
        try:
            for i in range(10):
                add(store, f"w{n}-{i}.md", [f"text {n} {i}"], vecs([0, 1, 0]))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def reader():
        try:
            for _ in range(30):
                store.vector_search(np.asarray([1, 0, 0], dtype=np.float32), 5)
                store.fts_search("text", 5)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and store.count_chunks() == 31


# ---- embedders -------------------------------------------------------------------------------

def test_hashing_embedder_properties():
    e = HashingEmbedder()
    a, b, c = e.embed(["annual leave days for employees", "employees annual leave days", "database index tuning"])
    assert a.shape == (2048,) and a.dtype == np.float32
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert float(a @ b) > 0.8 and float(a @ c) < 0.2
    assert np.array_equal(e.embed(["same text"]), e.embed(["same text"]))  # deterministic across calls
    z = e.embed(["", "the and of"])  # no content terms: zero vector, not NaN
    assert not np.isnan(z).any() and np.allclose(z, 0)
    assert e.embed([]).shape == (0, 2048)


def test_hashing_embedder_is_deterministic_across_processes():
    import subprocess, sys
    code = "from rag_docqa.embedders import HashingEmbedder as H; print(H().embed(['annual leave'])[0].nonzero()[0].tolist())"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".").stdout for _ in range(2)}
    assert len(outs) == 1 and outs != {""}  # PYTHONHASHSEED must not leak into stored vectors


def test_make_embedder_accepts_specs_and_names():
    assert make_embedder("hashing").dim == 2048
    assert make_embedder("hashing:512").dim == 512
    assert make_embedder("hashing-4096").dim == 4096  # the form an index stores
    e = HashingEmbedder(1024)
    assert make_embedder(e.name).name == e.name
    for bad in ("nope", "hashing-x", "hashing:abc"):
        with pytest.raises(ValueError):
            make_embedder(bad)


class _EmbedHandler(BaseHTTPRequestHandler):
    calls: list = []
    fail = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _EmbedHandler.calls.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if _EmbedHandler.fail:
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error":"rate limited"}')
            return
        inputs = body["input"]
        data = [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(inputs)]
        payload = json.dumps({"data": list(reversed(data))}).encode()  # out of order on purpose
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def embed_server(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    _EmbedHandler.calls, _EmbedHandler.fail = [], False
    httpd = HTTPServer(("127.0.0.1", 0), _EmbedHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_openai_compatible_embedder_batches_orders_and_normalises(embed_server):
    e = OpenAICompatibleEmbedder("m", api_key="k", base_url=embed_server, batch_size=2)
    out = e.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert len(_EmbedHandler.calls) == 3  # 5 inputs in batches of 2
    assert _EmbedHandler.calls[0]["path"] == "/embeddings" and _EmbedHandler.calls[0]["auth"] == "Bearer k"
    assert _EmbedHandler.calls[0]["body"] == {"model": "m", "input": ["a", "bb"]}
    assert out.shape == (5, 3) and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert out[0][0] < out[4][0]  # rows came back in input order despite the server reversing them
    assert e.name == "openai:m"


def test_openai_compatible_embedder_prefixes_and_errors(embed_server, monkeypatch):
    e = OpenAICompatibleEmbedder("m", api_key="k", base_url=embed_server, query_prefix="query: ", document_prefix="passage: ")
    e.embed(["x"], kind="query")
    e.embed(["x"], kind="document")
    assert [c["body"]["input"][0] for c in _EmbedHandler.calls] == ["query: x", "passage: x"]
    _EmbedHandler.fail = True
    with pytest.raises(EmbeddingError, match="429"):
        e.embed(["x"])
    with pytest.raises(EmbeddingError):
        OpenAICompatibleEmbedder("m", api_key="k", base_url="http://127.0.0.1:9", timeout=1).embed(["x"])
    monkeypatch.delenv("EMBEDDINGS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="EMBEDDINGS_API_KEY"):
        OpenAICompatibleEmbedder("m")


# ---- retriever -------------------------------------------------------------------------------

def test_rrf_math_and_tie_breaking():
    scores = rrf([[10, 20, 30], [20, 10]], k=60)
    assert scores[10] == pytest.approx(1 / 61 + 1 / 62) and scores[20] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[30] == pytest.approx(1 / 63)
    assert rrf([], 60) == {} and rrf([[]], 60) == {}


def test_hybrid_surfaces_hits_found_by_only_one_retriever(store):
    # chunk 1 only matches by keyword ("zebra"); chunk 2 only matches by vector (query vector == [0,1,0]).
    add(store, "a.md", ["zebra crossing rules", "completely unrelated prose", "more filler words"],
        vecs([1, 0, 0], [0, 1, 0], [0, 0, 1]))
    r = Retriever(store, FakeEmbedder((0, 1, 0)))
    by_kw = r.search("zebra", 3, "keyword")
    by_vec = r.search("zebra", 3, "vector")
    hybrid = r.search("zebra", 2, "hybrid")
    assert [h.chunk.text for h in by_kw] == ["zebra crossing rules"]
    assert by_vec[0].chunk.text == "completely unrelated prose"
    assert {h.chunk.text for h in hybrid} == {"zebra crossing rules", "completely unrelated prose"}
    top = hybrid[0]
    assert top.keyword_rank is not None or top.vector_rank is not None


def test_modes_sources_and_edge_cases(retriever):
    q = "How many days of annual leave do employees get?"
    for mode in MODES:
        hits = retriever.search(q, 3, mode)
        assert hits and hits[0].chunk.source == "leave_policy.md"
    only = retriever.search(q, 5, "hybrid", sources=["expenses_policy.md"])
    assert only and all(h.chunk.source == "expenses_policy.md" for h in only)
    assert retriever.search(q, 5, sources=["missing.md"]) == []
    assert retriever.search("   ", 5) == []
    with pytest.raises(ValueError):
        retriever.search(q, 5, "semantic")
    assert len(retriever.search(q, 2)) == 2


def test_hits_carry_cosine_and_ranks(retriever):
    (top, *_) = retriever.search("meal allowance international", 3)
    assert 0 < top.cosine <= 1.0 and top.keyword_rank == 1 and top.vector_rank == 1 and top.score > 0


def test_retriever_on_empty_store(store, emb):
    assert Retriever(store, emb).search("anything", 5) == []
