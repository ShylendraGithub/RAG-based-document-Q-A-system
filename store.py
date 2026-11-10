"""SQLite document store: documents, chunks, an FTS5 keyword index and float32 vectors.

Everything lives in one file and one transaction per document, so the keyword index and the
vector data can never disagree about which chunks exist.

  documents    one row per source file (content hash -> cheap "unchanged?" checks on re-ingest)
  chunks       text + section + pages + embedding BLOB (float32, L2-normalised)
  chunks_fts   FTS5 (porter stemming) over title + section + text; rowid == chunks.id

Vector search is exact brute force: all embeddings are held in one numpy matrix and scored with a
single matrix-vector product. That is fast and simple up to roughly 100k chunks; beyond that you
would move to an ANN index (pgvector HNSW, FAISS, sqlite-vec). The Store API is the seam for that swap.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunker import Chunk
from .text import fts_query


class StoreError(Exception):
    pass


@dataclass
class StoredChunk:
    id: int
    doc_id: int
    source: str
    title: str
    ordinal: int
    text: str
    section: str | None
    page_start: int | None
    page_end: int | None

    @property
    def location(self) -> str:
        parts = []
        if self.section:
            parts.append(self.section)
        if self.page_start is not None:
            parts.append(
                f"p. {self.page_start}"
                if self.page_end in (None, self.page_start)
                else f"pp. {self.page_start}-{self.page_end}"
            )
        return ", ".join(parts)


@dataclass
class DocInfo:
    id: int
    source: str
    title: str
    sha256: str
    n_chunks: int
    ingested_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    n_chunks    INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    section    TEXT,
    page_start INTEGER,
    page_end   INTEGER,
    embedding  BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, tokenize = 'porter unicode61');
"""


class Store:
    def __init__(self, path: str | Path, embedder_name: str | None = None):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._cache: tuple[np.ndarray, np.ndarray, dict[int, int]] | None = None  # ids, matrix, id->row
        self.embedder_name = self._bind_embedder(embedder_name)

    # -- embedder compatibility ---------------------------------------------------------------
    def _bind_embedder(self, name: str | None) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
            stored = row[0] if row else None
            if name is None:
                return stored
            if stored is not None and stored != name:
                raise StoreError(
                    f"This index was built with embedder {stored!r}, not {name!r}. Vectors from different "
                    "models are not comparable: re-ingest into a new database (or delete this one)."
                )
            if stored is None:
                self._conn.execute("INSERT INTO meta (key, value) VALUES ('embedder', ?)", (name,))
                self._conn.commit()
            return name

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes -------------------------------------------------------------------------------
    def add_document(
        self, source: str, title: str, sha256: str, chunks: list[Chunk], vectors: np.ndarray
    ) -> int:
        """Insert (or atomically replace) a document with its chunks and vectors."""
        if len(chunks) != len(vectors):
            raise StoreError("chunks and vectors must have the same length")
        if len(chunks) == 0:
            raise StoreError("Refusing to index a document with no chunks")
        dim = vectors.shape[1]
        with self._lock:
            known = self._conn.execute("SELECT length(embedding) FROM chunks LIMIT 1").fetchone()
            if known and known[0] != dim * 4:
                raise StoreError(f"Vector size {dim} does not match the vectors already stored ({known[0] // 4})")
            try:
                with self._conn:  # one transaction: all of it or none of it
                    self._delete(source)
                    cur = self._conn.execute(
                        "INSERT INTO documents (source, title, sha256, n_chunks, ingested_at) VALUES (?,?,?,?,?)",
                        (source, title, sha256, len(chunks), datetime.now(timezone.utc).isoformat(timespec="seconds")),
                    )
                    doc_id = cur.lastrowid
                    for chunk, vec in zip(chunks, vectors):
                        c = self._conn.execute(
                            "INSERT INTO chunks (doc_id, ordinal, text, section, page_start, page_end, embedding) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (doc_id, chunk.ordinal, chunk.text, chunk.section, chunk.page_start, chunk.page_end,
                             np.asarray(vec, dtype="<f4").tobytes()),
                        )
                        self._conn.execute(
                            "INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
                            (c.lastrowid, chunk.index_text(title)),
                        )
            finally:
                self._cache = None
            return doc_id

    def _delete(self, source: str) -> bool:
        row = self._conn.execute("SELECT id FROM documents WHERE source = ?", (source,)).fetchone()
        if not row:
            return False
        self._conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?)", (row[0],)
        )
        self._conn.execute("DELETE FROM documents WHERE id = ?", (row[0],))  # chunks cascade
        return True

    def delete_document(self, source: str) -> bool:
        with self._lock:
            try:
                with self._conn:
                    deleted = self._delete(source)
            finally:
                self._cache = None
            return deleted

    # -- reads --------------------------------------------------------------------------------
    def documents(self) -> list[DocInfo]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, source, title, sha256, n_chunks, ingested_at FROM documents ORDER BY source"
            ).fetchall()
        return [DocInfo(*r) for r in rows]

    def get_hash(self, source: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT sha256 FROM documents WHERE source = ?", (source,)).fetchone()
        return row[0] if row else None

    def resolve_sources(self, sources: list[str]) -> list[int]:
        with self._lock:
            marks = ",".join("?" * len(sources))
            rows = self._conn.execute(
                f"SELECT id FROM documents WHERE source IN ({marks})", list(sources)
            ).fetchall() if sources else []
        return [r[0] for r in rows]

    def count_chunks(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_chunks(self, ids: list[int]) -> dict[int, StoredChunk]:
        if not ids:
            return {}
        with self._lock:
            marks = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT c.id, c.doc_id, d.source, d.title, c.ordinal, c.text, c.section, c.page_start, c.page_end "
                f"FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({marks})",
                list(ids),
            ).fetchall()
        return {r[0]: StoredChunk(*r) for r in rows}

    def fts_search(self, query: str, limit: int, doc_ids: list[int] | None = None) -> list[int]:
        """Chunk ids ranked by BM25 (best first). ``query`` is free text; it is made FTS-safe here."""
        match = fts_query(query)
        if match is None:
            return []
        sql = "SELECT chunks_fts.rowid FROM chunks_fts"
        params: list = [match]
        if doc_ids is not None:
            sql += " JOIN chunks c ON c.id = chunks_fts.rowid"
        sql += " WHERE chunks_fts MATCH ?"
        if doc_ids is not None:
            sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            params += list(doc_ids)
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        params.append(limit)
        with self._lock:
            return [r[0] for r in self._conn.execute(sql, params).fetchall()]

    # -- vectors ------------------------------------------------------------------------------
    def _load_matrix(self) -> tuple[np.ndarray, np.ndarray, dict[int, int], np.ndarray]:
        with self._lock:
            if self._cache is None:
                rows = self._conn.execute("SELECT id, doc_id, embedding FROM chunks ORDER BY id").fetchall()
                if rows:
                    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
                    docs = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
                    matrix = np.vstack([np.frombuffer(r[2], dtype="<f4") for r in rows]).astype(np.float32)
                else:
                    ids = docs = np.zeros(0, dtype=np.int64)
                    matrix = np.zeros((0, 0), dtype=np.float32)
                self._cache = (ids, matrix, {int(i): n for n, i in enumerate(ids)}, docs)
            return self._cache  # type: ignore[return-value]

    def vector_search(
        self, query_vec: np.ndarray, limit: int, doc_ids: list[int] | None = None
    ) -> list[tuple[int, float]]:
        """Top ``limit`` (chunk_id, cosine) pairs, best first. Vectors are normalised, so dot == cosine."""
        ids, matrix, _, docs = self._load_matrix()
        if len(ids) == 0:
            return []
        if matrix.shape[1] != query_vec.shape[0]:
            raise StoreError(f"Query vector has {query_vec.shape[0]} dims but the index has {matrix.shape[1]}")
        scores = matrix @ query_vec.astype(np.float32)
        if doc_ids is not None:
            scores = np.where(np.isin(docs, np.asarray(doc_ids, dtype=np.int64)), scores, -np.inf)
        k = min(limit, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def cosines(self, chunk_ids: list[int], query_vec: np.ndarray) -> dict[int, float]:
        ids, matrix, index, _ = self._load_matrix()
        out: dict[int, float] = {}
        for cid in chunk_ids:
            row = index.get(cid)
            if row is not None:
                out[cid] = float(matrix[row] @ query_vec)
        return out
