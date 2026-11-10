"""Ingestion: load -> chunk -> embed -> store, incrementally.

* unchanged files (same SHA-256) are skipped, so re-running ingest on a folder is cheap
* changed files replace their old chunks atomically
* one bad file (corrupt PDF, embedding outage) is reported and does not abort the batch
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunker import chunk_document
from .embedders import Embedder, EmbeddingError
from .loaders import LoaderError, find_documents, load_document
from .store import Store, StoreError


@dataclass
class IngestResult:
    source: str
    status: str  # added | updated | unchanged | failed
    n_chunks: int = 0
    message: str | None = None


def ingest_path(
    path: str | Path,
    store: Store,
    embedder: Embedder,
    *,
    max_words: int = 180,
    overlap_words: int = 30,
    force: bool = False,
    progress: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    results: list[IngestResult] = []

    def record(r: IngestResult) -> None:
        results.append(r)
        if progress:
            progress(r)

    try:
        files = find_documents(path)
    except LoaderError as exc:
        record(IngestResult(str(path), "failed", message=str(exc)))
        return results
    if not files:
        record(IngestResult(str(path), "failed", message="No supported documents found"))
        return results

    for file_path, source in files:
        try:
            doc = load_document(file_path, source)
            existing = store.get_hash(source)
            if existing == doc.sha256 and not force:
                record(IngestResult(source, "unchanged"))
                continue
            chunks = chunk_document(doc.segments, max_words, overlap_words)
            if not chunks:
                raise LoaderError("No text left after chunking")
            vectors = embedder.embed([c.index_text(doc.title) for c in chunks], kind="document")
            store.add_document(source, doc.title, doc.sha256, chunks, vectors)
            record(IngestResult(source, "updated" if existing else "added", len(chunks)))
        except (LoaderError, EmbeddingError, StoreError, ValueError) as exc:
            record(IngestResult(source, "failed", message=str(exc)))
    return results
