import http.client
import json
import threading

import pytest

from rag_docqa import cli
from rag_docqa.assistant import RAGAssistant
from rag_docqa.demo_llm import DemoLLM
from rag_docqa.evaluate import answer_report, format_retrieval_table, retrieval_report
from rag_docqa.llm import AnthropicClient, LLMError
from rag_docqa.retriever import Retriever
from rag_docqa.server import PAGE, make_server

from conftest import SAMPLE_DOCS, ScriptedLLM

import json as _json
from http.server import BaseHTTPRequestHandler, HTTPServer


# ---- demo model, end to end ------------------------------------------------------------------

@pytest.mark.parametrize(
    "question, source, expected",
    [
        ("How many days of annual leave do full-time employees get?", "leave_policy.md", "24 days"),
        ("When do carried-over days expire?", "leave_policy.md", "31 March"),
        ("What is the daily meal allowance for international travel?", "expenses_policy.md", "$85"),
        ("What is the hotel cap per night in London?", "expenses_policy.md", "$320"),
        ("How long must passwords be?", "security_policy.md", "14 characters"),
        ("How quickly must I report a stolen laptop?", "security_policy.md", "2 hours"),
        ("How long is the probation period?", "onboarding_guide.md", "6 months"),
        ("Where is the main office located?", "onboarding_guide.md", "Pune"),
    ],
)
def test_demo_pipeline_answers_with_the_right_source(retriever, question, source, expected):
    a = RAGAssistant(retriever, DemoLLM()).ask(question)
    assert a.status == "ok" and expected in a.text
    assert a.citations and a.citations[0].source == source and not a.invalid_citations


@pytest.mark.parametrize("question", ["What is the company stock ticker symbol?", "What is the dental insurance coverage?"])
def test_demo_pipeline_declines_out_of_scope_questions(retriever, question):
    assert RAGAssistant(retriever, DemoLLM()).ask(question).status in ("no_context", "unanswerable")


def test_retrieval_report_is_correct_on_the_bundled_questions(retriever):
    questions = json.loads((SAMPLE_DOCS.parent / "eval" / "questions.json").read_text())
    report = retrieval_report(retriever, questions, "hybrid")
    assert report["n"] == 19 and report["hit@5"] >= 0.9 and 0 < report["mrr"] <= 1
    table = format_retrieval_table([report])
    assert "hybrid" in table and "MRR" in table


def test_retrieval_report_counts_misses_and_ranks(store, emb):
    from rag_docqa.chunker import Chunk
    import numpy as np
    store.add_document("a.md", "T", "h", [Chunk(0, "alpha beta"), Chunk(1, "gamma delta")],
                       emb.embed(["alpha beta", "gamma delta"]))
    r = Retriever(store, emb)
    qs = [{"question": "alpha", "source": "a.md", "phrase": "alpha beta"},
          {"question": "gamma", "source": "a.md", "phrase": "does not exist"},
          {"question": "x", "expect": "no_answer"}]
    rep = retrieval_report(r, qs, "keyword")
    assert rep["n"] == 2 and rep["hit@1"] == 0.5 and rep["mrr"] == 0.5 and rep["misses"] == ["gamma"]


def test_answer_report_scores_correctness_citations_and_refusals(retriever):
    qs = [
        {"question": "How many days of annual leave do employees get?", "source": "leave_policy.md", "answer_contains": ["24"]},
        {"question": "How many days of annual leave do employees get?", "source": "leave_policy.md", "answer_contains": ["24"]},
        {"question": "What is the dental insurance coverage?", "expect": "no_answer"},
        {"question": "How many days of annual leave do employees get?", "source": "expenses_policy.md", "answer_contains": ["24"]},
    ]
    # The out-of-scope question is stopped by the answerability gate and never reaches the LLM,
    # so only three replies are consumed.
    llm = ScriptedLLM("24 days [1]", "It is 30 days [1]", "24 days [1]")
    rep = answer_report(RAGAssistant(retriever, llm), qs)
    assert rep["answerable"] == 3 and rep["answer_correct"] == pytest.approx(2 / 3, abs=0.001)
    assert rep["cites_gold_source"] == pytest.approx(2 / 3, abs=0.001)
    assert rep["citations_valid"] == 1.0 and rep["refusal_correct"] == 1.0


# ---- server ----------------------------------------------------------------------------------

@pytest.fixture
def server(indexed, emb, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    assistant = RAGAssistant(Retriever(indexed, emb), DemoLLM())
    httpd = make_server(assistant, indexed, "127.0.0.1", 0, demo=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), data


def ask(port, payload, **kw):
    body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    return request(port, "POST", "/api/ask", body, {"Content-Type": "application/json", **kw.get("headers", {})})


def test_server_serves_ui_info_and_answers(server):
    status, headers, data = request(server, "GET", "/")
    assert status == 200 and b"Document Q&amp;A" in data and "default-src 'none'" in headers["Content-Security-Policy"]
    status, _, data = request(server, "GET", "/api/info")
    info = json.loads(data)
    assert status == 200 and info["demo"] is True and len(info["documents"]) == 4
    status, _, data = ask(server, {"question": "How long is the probation period?"})
    out = json.loads(data)
    assert status == 200 and out["status"] == "ok" and "6 months" in out["answer"]
    assert out["citations"][0]["source"] == "onboarding_guide.md" and out["passages"][0]["n"] == 1
    status, _, data = ask(server, {"question": "Who is the CEO?", "mode": "keyword"})
    assert json.loads(data)["status"] in ("no_context", "unanswerable")
    status, _, data = ask(server, {"question": "hotel cap", "sources": ["expenses_policy.md"]})
    assert {p["source"] for p in json.loads(data)["passages"]} == {"expenses_policy.md"}


@pytest.mark.parametrize(
    "payload, code",
    [
        ("not json", 400), ("{}", 400), ('{"question": ""}', 400), ('{"question": 5}', 400),
        ('{"question": "x", "mode": "magic"}', 400), ('{"question": "x", "sources": "a.md"}', 400),
        (json.dumps({"question": "x" * 1001}), 400),
    ],
)
def test_server_rejects_bad_requests(server, payload, code):
    assert ask(server, payload)[0] == code


def test_server_http_hardening(server):
    assert request(server, "POST", "/api/ask", "{}", {"Content-Type": "text/plain"})[0] == 415  # simple cross-site form
    assert request(server, "POST", "/api/ask", "x" * 20000, {"Content-Type": "application/json"})[0] == 413
    assert request(server, "GET", "/nope")[0] == 404
    assert request(server, "POST", "/other", "{}", {"Content-Type": "application/json"})[0] == 404
    assert request(server, "GET", "/api/info", headers={"Host": "evil.example"})[0] == 403  # DNS rebinding
    assert ask(server, {"question": "x"}, headers={"Host": "evil.example"})[0] == 403
    assert request(server, "GET", "/api/info", headers={"Host": f"localhost:{server}"})[0] == 200


def test_ui_never_inserts_untrusted_text_as_html():
    assert "innerHTML" not in PAGE and "insertAdjacentHTML" not in PAGE and "document.write" not in PAGE
    assert "textContent" in PAGE


# ---- CLI -------------------------------------------------------------------------------------

def run_cli(capsys, *argv):
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_cli_demo_ask_and_listing(tmp_path, capsys):
    db = str(tmp_path / "d.db")
    code, out, err = run_cli(capsys, "--demo", "--db", db, "ask", "How long is the probation period?", "--show-passages")
    assert code == 0 and "6 months" in out and "Sources:" in out and "onboarding_guide.md" in out
    assert "Retrieved passages" in out and "indexing the sample documents" in err
    code, out, _ = run_cli(capsys, "--db", db, "list")
    assert code == 0 and "leave_policy.md" in out and "chunks" in out
    code, out, _ = run_cli(capsys, "--demo", "--db", db, "ask", "Who is the CEO?")
    assert code == 0 and "I don't know" in out


def test_cli_ingest_delete_and_exit_codes(tmp_path, capsys):
    db = str(tmp_path / "x.db")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nPlain text about widgets and gadgets.")
    code, out, _ = run_cli(capsys, "--db", db, "ingest", str(docs))
    assert code == 0 and "added" in out and "1 document(s)" in out
    code, out, _ = run_cli(capsys, "--db", db, "ingest", str(docs))
    assert code == 0 and "unchanged" in out
    code, out, _ = run_cli(capsys, "--db", db, "ingest", str(tmp_path / "missing"))
    assert code == 1 and "failed" in out
    assert run_cli(capsys, "--db", db, "delete", "a.md")[0] == 0
    assert run_cli(capsys, "--db", db, "delete", "a.md")[0] == 1
    assert "empty" in run_cli(capsys, "--db", db, "list")[1]


def test_cli_embedder_mismatch_and_missing_api_key(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "m.db")
    run_cli(capsys, "--demo", "--db", db, "ask", "x")
    code, _, err = run_cli(capsys, "--db", db, "--embedder", "hashing:512", "list")
    assert code == 2 and "built with embedder" in err
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, _, err = run_cli(capsys, "--db", db, "ask", "anything")
    assert code == 2 and "ANTHROPIC_API_KEY" in err


# ---- real LLM client against a local stub (no live API call) -------------------------------------

class _Stub(BaseHTTPRequestHandler):
    captured: dict = {}
    status = 200

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        _Stub.captured = {"path": self.path, "key": self.headers.get("x-api-key"), "body": _json.loads(self.rfile.read(n))}
        payload = _json.dumps({"content": [{"type": "text", "text": "Answer [1]."}]}).encode()
        self.send_response(_Stub.status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def test_anthropic_client_request_shape_and_errors(monkeypatch):
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    _Stub.status = 200
    httpd = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = AnthropicClient(api_key="k", model="m", base_url=f"http://127.0.0.1:{httpd.server_address[1]}")
        assert client.complete("sys", [{"role": "user", "content": "hi"}]) == "Answer [1]."
        assert _Stub.captured["path"] == "/v1/messages" and _Stub.captured["key"] == "k"
        assert _Stub.captured["body"] == {"model": "m", "max_tokens": 1024, "system": "sys",
                                          "messages": [{"role": "user", "content": "hi"}]}
        _Stub.status = 500
        with pytest.raises(LLMError, match="500"):
            client.complete("sys", [{"role": "user", "content": "hi"}])
    finally:
        httpd.shutdown()
        httpd.server_close()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()
