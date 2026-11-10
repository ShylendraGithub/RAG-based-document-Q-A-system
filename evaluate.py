"""Evaluation harness: measure retrieval and answers instead of eyeballing them.

Questions file (JSON list). Answerable items carry the gold location:
    {"question": ..., "source": "leave_policy.md", "phrase": "24 days", "answer_contains": ["24"]}
  * a retrieved chunk is a HIT when it comes from ``source`` and contains ``phrase``
  * ``answer_contains``: for --answers, the generated answer must contain one of these (case-insensitive)
Unanswerable items:  {"question": ..., "expect": "no_answer"}

Retrieval metrics (answerable items only), computed for keyword / vector / hybrid so you can see
what each retriever contributes:
  hit@k   fraction of questions with a gold chunk in the top k
  MRR     mean of 1/rank of the first gold chunk (0 if none in the top 10)

Answer metrics (--answers, needs an LLM):
  answer_correct      answer contains an expected string and the status is ok
  cites_gold_source   a cited passage comes from the gold source
  citations_valid     at least one citation, none pointing at a passage that wasn't supplied
  refusal_correct     unanswerable questions end as no_context / unanswerable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assistant import RAGAssistant
from .demo_llm import DemoLLM
from .embedders import EmbeddingError, make_embedder
from .llm import AnthropicClient, LLMError
from .retriever import MODES, Retriever
from .store import Store, StoreError

DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.json"
RANK_DEPTH = 10


def _is_gold(item: dict, chunk) -> bool:
    return chunk.source == item["source"] and item["phrase"].lower() in " ".join(chunk.text.split()).lower()


def retrieval_report(retriever: Retriever, questions: list[dict], mode: str, ks=(1, 3, 5)) -> dict:
    answerable = [q for q in questions if "phrase" in q]
    ranks: list[int | None] = []
    for q in answerable:
        hits = retriever.search(q["question"], RANK_DEPTH, mode)
        rank = next((i for i, h in enumerate(hits, start=1) if _is_gold(q, h.chunk)), None)
        ranks.append(rank)
    n = len(answerable)
    report = {"mode": mode, "n": n}
    for k in ks:
        report[f"hit@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 3) if n else None
    report["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3) if n else None
    report["misses"] = [q["question"] for q, r in zip(answerable, ranks) if r is None]
    return report


def answer_report(assistant: RAGAssistant, questions: list[dict]) -> dict:
    correct = gold_cited = valid = answerable = refusals = refusals_ok = 0
    details = []
    for q in questions:
        a = assistant.ask(q["question"])
        if q.get("expect") == "no_answer":
            refusals += 1
            ok = a.status in ("no_context", "unanswerable")
            refusals_ok += ok
            details.append({"question": q["question"], "passed": ok, "status": a.status})
            continue
        answerable += 1
        text = a.text.lower()
        is_correct = a.status == "ok" and any(s.lower() in text for s in q.get("answer_contains", []))
        cites_gold = any(c.source == q["source"] for c in a.citations)
        cites_valid = a.status == "ok" and bool(a.citations) and not a.invalid_citations
        correct += is_correct
        gold_cited += cites_gold
        valid += cites_valid
        details.append({"question": q["question"], "passed": bool(is_correct), "status": a.status,
                        "cites_gold_source": cites_gold, "answer": a.text})
    return {
        "answerable": answerable,
        "answer_correct": round(correct / answerable, 3) if answerable else None,
        "cites_gold_source": round(gold_cited / answerable, 3) if answerable else None,
        "citations_valid": round(valid / answerable, 3) if answerable else None,
        "refusal_cases": refusals,
        "refusal_correct": round(refusals_ok / refusals, 3) if refusals else None,
        "details": details,
    }


def format_retrieval_table(reports: list[dict]) -> str:
    head = f"{'mode':<9} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6}"
    rows = [f"{r['mode']:<9} {r['hit@1']:>6.2f} {r['hit@3']:>6.2f} {r['hit@5']:>6.2f} {r['mrr']:>6.2f}" for r in reports]
    return "\n".join([head, "-" * len(head)] + rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag_docqa.evaluate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--embedder", default=None, help="must match the index (default: the one stored in the db)")
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    ap.add_argument("--answers", action="store_true", help="also score generated answers (needs an LLM)")
    ap.add_argument("--demo", action="store_true", help="use the offline extractive stand-in instead of an LLM")
    ap.add_argument("--model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        probe = Store(args.db)
        stored = probe.embedder_name
        probe.close()
        embedder = make_embedder(args.embedder or stored or "hashing")
        store = Store(args.db, embedder.name)
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        retriever = Retriever(store, embedder)
        reports = [retrieval_report(retriever, questions, m) for m in ("keyword", "vector", "hybrid")]
        out: dict = {"embedder": embedder.name, "retrieval": reports}
        if args.answers:
            llm = DemoLLM() if args.demo else AnthropicClient(model=args.model)
            out["answers"] = answer_report(RAGAssistant(retriever, llm), questions)
    except (StoreError, EmbeddingError, LLMError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"Embedder: {embedder.name}   questions: {reports[0]['n']} answerable\n")
    print(format_retrieval_table(reports))
    for r in reports:
        if r["misses"]:
            print(f"\n{r['mode']} missed (top {RANK_DEPTH}): " + "; ".join(r["misses"]))
    if "answers" in out:
        a = out["answers"]
        print(
            f"\nAnswers: correct {a['answer_correct']:.2f} | cites gold source {a['cites_gold_source']:.2f} | "
            f"citations valid {a['citations_valid']:.2f} | refusals handled {a['refusal_correct']:.2f}"
        )
        for d in a["details"]:
            if not d["passed"]:
                print(f"  [FAIL] {d['question']} -> {d['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
