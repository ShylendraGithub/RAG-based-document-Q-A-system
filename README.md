# RAG Document Q&A

Ask questions about your own documents (PDF, Word, Markdown, HTML, text) and get answers with
**citations you can verify**. Hybrid retrieval (BM25 keyword search + vector search) finds the
passages, an LLM answers only from them, and the pipeline checks that every citation points at a
passage that was really supplied.

```
$ python -m rag_docqa --demo ask "What is the daily meal allowance for international travel?"

The daily meal allowance is $60 when travelling domestically and $85 when travelling internationally. [1]

Sources:
  [1] expenses_policy.md (Travel and Expense Policy > Meal allowance)

$ python -m rag_docqa --demo ask "Who is the CEO?"
I don't know based on the provided documents.
[No sufficiently relevant passages were found, so the model was not asked.]
```

`--demo` runs fully offline with sample documents and a **rule-based stand-in for the LLM**
(see [Demo mode](#demo-mode)). `python -m rag_docqa --demo serve` opens a small web UI.

## How it works

```
 files ──► loaders ──► chunker ──► embedder ──► SQLite (one file)
 pdf docx   text +     sections,    vectors     ├ documents  (content hash → incremental re-ingest)
 md html    pages      overlap                  ├ chunks     (text, section, pages, float32 vector)
 txt                                            └ chunks_fts (FTS5 / BM25, porter stemming)

 question ─► embed ─┬─► vector search (cosine) ──┐
                    └─► keyword search (BM25) ───┴─► Reciprocal Rank Fusion ─► top-k passages
                                                                                   │
        answerability gate ◄── cosine below threshold? ◄───────────────────────────┘
              │ no relevant passages → "I don't know" (LLM never called)
              ▼
        grounded prompt (numbered <document> blocks) ─► LLM ─► citation check ─► answer + sources
```

### Design decisions

| Decision | Why |
|----------|-----|
| **Hybrid retrieval with RRF** | BM25 is exact (names, codes, "$320") but blind to paraphrase; vectors handle paraphrase but can miss exact tokens. RRF merges the two *rankings*, so BM25 scores and cosine similarities never need to be put on one scale. |
| **Structure-aware chunking** | Chunks follow headings (`Leave Policy > Annual leave`), never straddle sections, pack whole sentences, overlap a few sentences at the boundaries, and record PDF pages. A hard word cap is enforced, even for a 500-word "sentence". |
| **Title + section are indexed with each chunk** | A passage that says "24 days" is still found for "leave policy", because its heading is part of what is embedded and keyword-indexed. |
| **Incremental, atomic ingestion** | Files are keyed by SHA-256: unchanged files are skipped, changed files replace their old chunks in one transaction (a failed replace rolls back and keeps the old version). One corrupt file never aborts a batch. |
| **Embedder name stored in the index** | Vectors from different models are not comparable. Opening an index with a different embedder is refused with a clear error instead of returning quietly wrong results. |
| **FTS queries are built, not passed through** | User text is reduced to quoted terms joined with `OR`, so it can never be parsed as FTS5 syntax (`NEAR`, `col:`, stray quotes). Tested with 17 hostile inputs. |
| **SQLite for everything** | One file, no server. The `Store` class is the seam if you outgrow it (see [Limits](#limits-and-honest-caveats)). |

### Guardrails against wrong answers

| Failure mode | What the pipeline does |
|--------------|------------------------|
| Answering when the documents don't cover the question | **Answerability gate**: if no passage is similar enough, the LLM isn't called at all. The model is also told to reply with a fixed sentence when the passages lack the answer. |
| Invented or mis-numbered citations | Every `[n]` is checked against the passages actually supplied. Invalid numbers are reported and never become citations; an answer with no citation is flagged `uncited`. |
| Prompt injection hidden in a document | Passages are wrapped in `<document>` tags with closing tags neutralised and attributes escaped; the system prompt says document text is data, not instructions. This **reduces** the risk, it cannot remove it: see the caveats. |
| UI/XSS from document text | The web UI only uses `textContent`, never `innerHTML`; the server sends a strict CSP, checks the `Host` header (DNS rebinding), accepts JSON POSTs only, and caps request size. |

## Quick start

Python 3.10+. The core needs only **numpy**; PDF and DOCX support are optional.

```bash
pip install numpy pypdf python-docx          # pypdf / python-docx only if you ingest those formats

# 1. Offline demo: no API key, no network
python -m rag_docqa --demo ask "How many days of annual leave do full-time employees get?" --show-passages
python -m rag_docqa --demo serve             # web UI on http://127.0.0.1:8000

# 2. Your own documents with a real model
cp .env.example .env                         # add ANTHROPIC_API_KEY
python -m rag_docqa ingest ./my_docs         # re-run any time: only changed files are re-indexed
python -m rag_docqa ask "What is our refund policy?" --show-passages
python -m rag_docqa ask "..." --source handbook.pdf      # restrict to one file
python -m rag_docqa list                     # what is indexed
python -m rag_docqa delete handbook.pdf
```

### Real (semantic) embeddings

The default `hashing` embedder is offline and **lexical, not semantic** (see below). For real
semantic search, pick an embedder when you create the index:

```bash
# any OpenAI-compatible /v1/embeddings endpoint
export EMBEDDINGS_API_KEY=...   # optional: EMBEDDINGS_BASE_URL=https://.../v1
python -m rag_docqa --embedder openai:text-embedding-3-small --db rag.db ingest ./my_docs

# or a local model
pip install sentence-transformers
python -m rag_docqa --embedder st:sentence-transformers/all-MiniLM-L6-v2 --db rag.db ingest ./my_docs
```

> **Status:** the OpenAI-compatible client is tested against a local stub server (request shape,
> batching, ordering, errors) but **never against a live service**. The sentence-transformers
> wrapper has **never been run** (the library was not installed where this was built). The
> answerability gate has no default threshold for these embedders (thresholds are model-specific):
> tune `--min-score` with the evaluation harness.

### As a library

```python
from rag_docqa import Store, Retriever, RAGAssistant, AnthropicClient, ingest_path, make_embedder

emb = make_embedder("hashing")
store = Store("rag.db", emb.name)
ingest_path("my_docs", store, emb)
bot = RAGAssistant(Retriever(store, emb), AnthropicClient())
a = bot.ask("What is our refund policy?")
print(a.status, a.text, [(c.source, c.location) for c in a.citations])
```

### Demo mode

`--demo` indexes the bundled sample documents (a fictional company's leave, expense, security and
onboarding policies) and uses `DemoLLM`, an **extractive stand-in, not an AI model**: it returns the
one or two sentences that share the most terms with the question, with their `[n]` citation. It
cannot paraphrase, combine facts or reason. Likewise `HashingEmbedder` matches stemmed words and
bigrams; it does not know that "holiday" means "vacation". Both exist so the whole system runs and
is testable offline.

## Evaluation

```bash
python -m rag_docqa.evaluate --db rag.db                    # retrieval metrics, no LLM needed
python -m rag_docqa.evaluate --db rag.db --answers          # + answer quality (needs ANTHROPIC_API_KEY)
```

`eval/questions.json` has 19 answerable questions (with the exact passage that should be found) and
4 that the documents cannot answer. The harness reports hit@1/3/5 and MRR for keyword, vector and
hybrid retrieval, plus answer correctness, citation validity and refusal handling.

Results of the bundled set with the offline components (**not** a real model's numbers):

```
mode       hit@1  hit@3  hit@5    MRR
keyword     1.00   1.00   1.00   1.00
vector      1.00   1.00   1.00   1.00
hybrid      1.00   1.00   1.00   1.00

Stand-in answers: correct 0.90 | cites gold source 0.95 | citations valid 0.95 | refusals handled 0.75
```

How to read this honestly:

* **All three retrievers tie at 1.00** because the sample corpus is tiny (4 documents, 17 chunks)
  and the offline embedder is lexical. This table **cannot show that hybrid beats keyword**. Run the
  harness on a few hundred of your own documents with a real embedder to get a meaningful comparison.
* The three stand-in failures are the stand-in's limits, not the pipeline's: it can't map "doctor's
  note" to "medical certificate" or "work from home" to "remotely", and it answers "Who is the CEO of
  Lumen Analytics?" with the company-introduction sentence because "Lumen Analytics" overlaps.
  A real LLM is expected to handle all three; measure that with `--answers` before claiming it.

## Performance (measured here, synthetic data)

20,000 chunks of 120 words, 2048-dim vectors, in-memory SQLite, one CPU:

| | |
|---|---|
| Ingest (hash-embed + insert) | ~4,100 chunks/s |
| Keyword / vector / hybrid query | 1.5 ms / 7.7 ms / 9.1 ms |
| Vector matrix in RAM | 164 MB (rebuilt in ~190 ms after any ingest or delete) |

These are the costs of exact brute-force search with a cheap embedder. With a remote embedding API,
embedding time and cost dominate ingestion, and query latency gains the API round trip.

## Tests

```bash
pip install pytest pypdf python-docx && python -m pytest        # 129 tests, ~6 s
```

* chunker: section paths, no chunk over the word limit, every word covered, overlap, page tracking, degenerate input
* loaders: txt/md/html/**real PDF bytes**/docx (headings, bullets, tables), scanned and corrupt files, size limit
* store: replace/delete across all three indexes, **rollback keeps the old version**, embedder binding, concurrent reads/writes
* retrieval: RRF math, hybrid surfaces hits only one retriever finds, source filters, FTS-injection corpus
* assistant: citation verification, hallucinated `[9]`, refusals, gate skips the LLM, prompt-injection escaping, failure paths
* server: API validation, Host/Content-Type/size hardening, UI has no `innerHTML`; CLI exit codes
* a headless-browser check (run manually, not part of the suite) confirmed the UI renders and that typing `<img onerror=...>` into the question box injects nothing

## Limits and honest caveats

* **Brute-force vector search.** Fine to roughly 100k chunks; beyond that use an ANN index
  (pgvector HNSW, FAISS, sqlite-vec). `Store.vector_search` is the single place to change.
* **No reranker, no conversation memory.** Each question is answered independently; follow-ups like
  "and for international?" need the previous turn rewritten into a standalone question first.
* **Prompt injection is reduced, not solved.** A document can still try to steer the model. For
  untrusted corpora add output review and keep the model's powers minimal (this one has no tools).
* **No access control.** All indexed documents are visible to every question. The web server has no
  authentication and binds to localhost: do not expose it as is.
* **PDF fidelity.** Text extraction flattens tables and multi-column layouts; scanned PDFs are
  rejected (no OCR).
* **Chunk defaults (180 words, 30 overlap) are reasonable starting values, not tuned** on a real
  corpus. Tune them with the evaluation harness.
* Answers sent to a hosted LLM include the retrieved passages: don't index documents you may not
  send to that provider.

## Project layout

```
rag_docqa/
  loaders.py     pdf / docx / html / md / txt → text segments (+ title, content hash)
  chunker.py     structure-aware chunking
  embedders.py   hashing (offline) · OpenAI-compatible HTTP · sentence-transformers
  store.py       SQLite: documents, chunks, FTS5 index, vectors
  retriever.py   keyword / vector / hybrid (RRF)
  ingest.py      incremental ingestion
  assistant.py   gate → prompt → LLM → citation verification
  llm.py         LLM protocol + Anthropic HTTP client (standard library)
  demo_llm.py    offline extractive stand-in for the LLM
  evaluate.py    retrieval + answer evaluation harness
  server.py      local web UI + JSON API
  cli.py         command line
sample_docs/  eval/  tests/
```

## Resume bullets

Fill the brackets with **your own measured numbers** (run `rag_docqa.evaluate` on your documents
with a real embedder and model; don't quote figures you haven't measured):

* Built a retrieval-augmented Q&A system in Python over PDF/DOCX/Markdown/HTML: structure-aware
  chunking, **hybrid BM25 + vector retrieval with Reciprocal Rank Fusion** on SQLite (FTS5), and
  grounded LLM answers with citations verified against the retrieved passages.
* Designed answer-safety guardrails (answerability gate that skips the LLM, citation verification,
  prompt-injection-aware prompt construction, hardened local web UI) and an evaluation harness
  reporting hit@k/MRR and answer correctness; hybrid retrieval reached **[X]** hit@3 vs **[Y]** for
  keyword-only on a **[N]**-question set.
* Implemented incremental, transactional ingestion (content hashing, atomic replace with rollback,
  embedder-compatibility checks) covered by a 129-test suite, including real-PDF and concurrency tests.

**Interview talking points:** why hybrid beats either retriever alone and why RRF avoids score
calibration; chunk size vs recall/precision; why the answerability gate is better than asking the
LLM to "be careful"; why prompt injection can't be fully prevented by prompting; brute-force vs ANN
trade-offs; why the evaluation says "all tie at 1.00" on the sample set and what you'd do about it.
