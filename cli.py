"""Command line interface.

    python -m rag_docqa --demo ask "How many days of annual leave do I get?"
    python -m rag_docqa --demo serve                       # web UI on http://127.0.0.1:8000

    python -m rag_docqa ingest ./my_docs                   # .pdf .docx .md .txt .html (re-runs are incremental)
    python -m rag_docqa ask "What is our refund policy?" --show-passages
    python -m rag_docqa list | delete <source> | serve
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .assistant import Answer, RAGAssistant
from .demo_llm import DemoLLM
from .embedders import EmbeddingError, make_embedder
from .ingest import IngestResult, ingest_path
from .llm import AnthropicClient, LLMError
from .retriever import MODES, Retriever
from .store import Store, StoreError

DEMO_DB = "demo_rag.db"
SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag_docqa", description="Ask questions about your documents, with citations.")
    p.add_argument("--db", help=f"index file (default: $RAG_DB or rag.db; {DEMO_DB} with --demo)")
    p.add_argument("--embedder", help="hashing | hashing:4096 | openai:<model> | st:<model> (default: the index's own)")
    p.add_argument("--demo", action="store_true", help="offline mode: sample documents + a rule-based stand-in for the LLM")
    p.add_argument("--model", help="LLM model id (default: $RAG_MODEL or the built-in default)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="index files or folders")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--max-words", type=int, default=180, help="max words per chunk")
    ing.add_argument("--overlap-words", type=int, default=30)
    ing.add_argument("--force", action="store_true", help="re-index unchanged files too")

    ask = sub.add_parser("ask", help="ask a question (omit it for an interactive prompt)")
    ask.add_argument("question", nargs="*")
    ask.add_argument("--source", action="append", help="restrict to this source file (repeatable)")
    ask.add_argument("--mode", choices=MODES, default="hybrid")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--show-passages", action="store_true", help="also print every retrieved passage")
    ask.add_argument("--min-score", type=float, help="cosine gate (default: the embedder's own; 0 disables)")

    sub.add_parser("list", help="list indexed documents")
    dele = sub.add_parser("delete", help="remove a document from the index")
    dele.add_argument("source")
    srv = sub.add_parser("serve", help="start the local web UI")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    return p


def _open(args) -> tuple[Store, object]:
    db = args.db or (DEMO_DB if args.demo else os.environ.get("RAG_DB", "rag.db"))
    stored = None
    if Path(db).exists():
        probe = Store(db)
        stored = probe.embedder_name
        probe.close()
    embedder = make_embedder(args.embedder or stored or "hashing")
    return Store(db, embedder.name), embedder


def _print_ingest(r: IngestResult) -> None:
    detail = f"({r.n_chunks} chunks)" if r.n_chunks else (f"- {r.message}" if r.message else "")
    print(f"  {r.status:<9} {r.source} {detail}".rstrip())


def print_answer(a: Answer, show_passages: bool = False) -> None:
    print(a.text if a.text else f"(no answer) {a.message}")
    if a.status == "no_context" and a.message:
        print(f"[{a.message}]")
    if a.status == "error":
        print(f"error: {a.message}")
    if a.invalid_citations:
        print(f"\nWARNING: the answer cites passage(s) {a.invalid_citations} that were not provided.")
    if a.status == "ok" and a.uncited:
        print("\nWARNING: the answer has no citations; verify it against the documents.")
    if a.citations:
        print("\nSources:")
        for c in a.citations:
            print(f"  [{c.n}] {c.source}" + (f" ({c.location})" if c.location else ""))
    if show_passages and a.hits:
        print("\nRetrieved passages:")
        for n, h in enumerate(a.hits, start=1):
            ranks = f"keyword #{h.keyword_rank or '-'}, vector #{h.vector_rank or '-'}, similarity {h.cosine:.2f}"
            print(f"  [{n}] {h.chunk.source} ({h.chunk.location or 'no section'}) - {ranks}")
            print("      " + " ".join(h.chunk.text.split())[:200])
    print(f"\n{a.elapsed_ms:g} ms")


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        store, embedder = _open(args)
    except (StoreError, EmbeddingError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.demo and not store.documents() and args.cmd != "ingest":
            print(f"[demo] indexing the sample documents into {store.path}", file=sys.stderr)
            for r in ingest_path(SAMPLE_DOCS, store, embedder):
                if r.status == "failed":
                    print(f"error: {r.source}: {r.message}", file=sys.stderr)
                    return 1

        if args.cmd == "ingest":
            failed = False
            for path in args.paths:
                for r in ingest_path(path, store, embedder, max_words=args.max_words,
                                     overlap_words=args.overlap_words, force=args.force, progress=_print_ingest):
                    failed |= r.status == "failed"
            print(f"\n{len(store.documents())} document(s), {store.count_chunks()} chunk(s) in {store.path}")
            return 1 if failed else 0

        if args.cmd == "list":
            docs = store.documents()
            if not docs:
                print("The index is empty. Run: python -m rag_docqa ingest <folder>")
            for d in docs:
                print(f"{d.source:<40} {d.n_chunks:>4} chunks   {d.title}")
            return 0

        if args.cmd == "delete":
            if store.delete_document(args.source):
                print(f"deleted {args.source}")
                return 0
            print(f"not found: {args.source}", file=sys.stderr)
            return 1

        llm = DemoLLM() if args.demo else AnthropicClient(model=args.model)
        if args.demo:
            print("[demo] offline rule-based stand-in in use; no LLM API call is made", file=sys.stderr)

        if args.cmd == "serve":
            from .server import make_server

            min_cosine = "auto"
            assistant = RAGAssistant(Retriever(store, embedder), llm, min_cosine=min_cosine)
            httpd = make_server(assistant, store, args.host, args.port, demo=args.demo)
            print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop; no authentication, keep it local)")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print()
            return 0

        # ask
        min_cosine = "auto" if args.min_score is None else (None if args.min_score == 0 else args.min_score)
        assistant = RAGAssistant(Retriever(store, embedder), llm, top_k=args.top_k, mode=args.mode, min_cosine=min_cosine)
        question = " ".join(args.question).strip()
        if question:
            answer = assistant.ask(question, sources=args.source)
            print_answer(answer, args.show_passages)
            return 0 if answer.status in ("ok", "unanswerable", "no_context") else 1
        print("Ask a question (type 'exit' to quit).")
        while True:
            try:
                question = input("\nrag> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in ("exit", "quit", "q"):
                return 0
            if question:
                print_answer(assistant.ask(question, sources=args.source), args.show_passages)
    except (LLMError, StoreError, EmbeddingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
