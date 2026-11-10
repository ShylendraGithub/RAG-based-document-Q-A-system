"""A tiny local web UI + JSON API (standard library only).

    GET  /               single-page UI
    GET  /api/info       {"demo": bool, "documents": [...], "embedder": ...}
    POST /api/ask        {"question": str, "mode"?: hybrid|keyword|vector, "sources"?: [str]}

Meant for local use. It binds to 127.0.0.1 by default and has NO authentication: do not expose it
to a network as is. Hardening that is included: loopback Host-header check (DNS-rebinding),
JSON-only POSTs (a cross-site form can't send them), a request-size cap, a strict CSP, and the UI
renders answers with textContent only (document text is untrusted, so it is never inserted as HTML).
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .assistant import Answer, RAGAssistant
from .retriever import MODES
from .store import Store

MAX_BODY = 16 * 1024
_LOOPBACK_HOST = re.compile(r"^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$", re.IGNORECASE)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Document Q&amp;A</title>
<style>
 :root{--bg:#fff;--fg:#1d2026;--mut:#5c6370;--line:#dde1e7;--acc:#2457d6;--card:#f6f7f9;--warn:#9a5b00}
 @media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8ec;--mut:#9aa1ad;--line:#2b2f37;--acc:#7aa2ff;--card:#1c1f25;--warn:#e0a458}}
 body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
 main{max-width:760px;margin:0 auto;padding:24px 16px 64px}
 h1{font-size:1.4rem;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 20px;font-size:.9rem}
 form{display:flex;gap:8px;flex-wrap:wrap} input[type=text]{flex:1 1 280px;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit}
 select,button{padding:10px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
 button{background:var(--acc);color:#fff;border-color:var(--acc);cursor:pointer} button:disabled{opacity:.6}
 .answer{margin-top:24px;padding:14px;border:1px solid var(--line);border-radius:10px;background:var(--card);white-space:pre-wrap}
 .note{margin-top:8px;color:var(--warn);font-size:.9rem} h2{font-size:1rem;margin:22px 0 6px}
 .src{border-left:3px solid var(--acc);padding:6px 10px;margin:8px 0;background:var(--card);border-radius:0 8px 8px 0;font-size:.9rem}
 .src b{font-weight:600} .meta{color:var(--mut);font-size:.8rem} details{margin-top:6px}
 #docs{color:var(--mut);font-size:.85rem;margin-top:28px}
</style></head><body><main>
<h1>Document Q&amp;A</h1><p class="sub" id="banner"></p>
<form id="f"><input type="text" id="q" placeholder="Ask a question about your documents" maxlength="1000" required autofocus>
<select id="mode" aria-label="Retrieval mode"><option>hybrid</option><option>keyword</option><option>vector</option></select>
<button id="go">Ask</button></form>
<div id="out"></div><div id="docs"></div>
<script>
const $=id=>document.getElementById(id);
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
fetch('/api/info').then(r=>r.json()).then(i=>{
 $('banner').textContent=(i.demo?'Demo mode: an offline rule-based stand-in picks sentences, it is not an AI model. ':'')+'Index: '+i.documents.length+' document(s), embedder '+i.embedder;
 $('docs').textContent='Documents: '+i.documents.map(d=>d.source+' ('+d.n_chunks+')').join(', ');
});
$('f').addEventListener('submit',async ev=>{
 ev.preventDefault();const out=$('out');out.replaceChildren();$('go').disabled=true;
 try{
  const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:$('q').value,mode:$('mode').value})});
  const d=await r.json();
  if(!r.ok){out.append(el('div','answer',d.error||'Request failed'));return}
  out.append(el('div','answer',d.answer||d.message||''));
  if(d.invalid_citations.length)out.append(el('div','note','Warning: the answer cites passage(s) that were not provided: '+d.invalid_citations.join(', ')));
  if(d.status==='ok'&&d.uncited)out.append(el('div','note','Warning: this answer contains no citations, treat it with caution.'));
  if(d.status==='no_context')out.append(el('div','note',d.message));
  if(d.citations.length){out.append(el('h2',null,'Sources'));
   d.citations.forEach(c=>{const s=el('div','src');s.append(el('b',null,'['+c.n+'] '+c.source));
    if(c.location)s.append(el('span','meta','  '+c.location));s.append(el('div',null,c.snippet));out.append(s)})}
  if(d.passages.length){const det=el('details');det.append(el('summary',null,'All retrieved passages ('+d.passages.length+')'));
   d.passages.forEach(p=>{const s=el('div','src');s.append(el('b',null,'['+p.n+'] '+p.source));
    s.append(el('span','meta','  similarity '+p.cosine.toFixed(2)+(p.keyword_rank?', keyword #'+p.keyword_rank:'')+(p.vector_rank?', vector #'+p.vector_rank:'')));
    s.append(el('div',null,p.text));det.append(s)});out.append(det)}
 }catch(e){out.append(el('div','answer','Request failed: '+e))}finally{$('go').disabled=false}
});
</script></main></body></html>
"""


def answer_to_json(a: Answer) -> dict:
    return {
        "question": a.question,
        "status": a.status,
        "answer": a.text,
        "message": a.message,
        "uncited": a.uncited,
        "invalid_citations": a.invalid_citations,
        "citations": [
            {"n": c.n, "source": c.source, "location": c.location, "snippet": c.snippet} for c in a.citations
        ],
        "passages": [
            {
                "n": n,
                "source": h.chunk.source,
                "location": h.chunk.location,
                "text": h.chunk.text,
                "cosine": round(h.cosine, 4),
                "keyword_rank": h.keyword_rank,
                "vector_rank": h.vector_rank,
            }
            for n, h in enumerate(a.hits, start=1)
        ],
        "elapsed_ms": a.elapsed_ms,
    }


def make_server(
    assistant: RAGAssistant, store: Store, host: str = "127.0.0.1", port: int = 8000, *, demo: bool = False
) -> ThreadingHTTPServer:
    loopback = host in ("127.0.0.1", "localhost", "::1")

    class Handler(BaseHTTPRequestHandler):
        server_version = "RagDocQA"

        def log_message(self, *args):  # keep test and demo output quiet
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            return not loopback or bool(_LOOPBACK_HOST.match(self.headers.get("Host", "")))

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "Forbidden host"})
            path = self.path.split("?", 1)[0]
            if path == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/info":
                docs = [{"source": d.source, "title": d.title, "n_chunks": d.n_chunks} for d in store.documents()]
                return self._json(200, {"demo": demo, "documents": docs, "embedder": store.embedder_name})
            self._json(404, {"error": "Not found"})

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "Forbidden host"})
            if self.path.split("?", 1)[0] != "/api/ask":
                return self._json(404, {"error": "Not found"})
            if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
                return self._json(415, {"error": "Content-Type must be application/json"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "Bad Content-Length"})
            if length <= 0 or length > MAX_BODY:
                return self._json(413 if length > MAX_BODY else 400, {"error": f"Body must be 1-{MAX_BODY} bytes"})
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                question = body["question"]
                mode = body.get("mode", "hybrid")
                sources = body.get("sources")
                if not isinstance(question, str) or not question.strip() or len(question) > 1000:
                    raise ValueError("question must be a non-empty string up to 1000 characters")
                if mode not in MODES:
                    raise ValueError(f"mode must be one of {list(MODES)}")
                if sources is not None and not (isinstance(sources, list) and all(isinstance(s, str) for s in sources)):
                    raise ValueError("sources must be a list of strings")
            except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
                return self._json(400, {"error": f"Invalid request: {exc}"})
            self._json(200, answer_to_json(assistant.ask(question, sources=sources, mode=mode)))

    return ThreadingHTTPServer((host, port), Handler)
