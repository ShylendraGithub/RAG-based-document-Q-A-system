"""Document loaders: .txt .md .rst .html .pdf .docx -> text segments (+ title, content hash).

PDF (pypdf) and DOCX (python-docx) are optional dependencies, imported lazily so the rest of
the package works without them. Scanned PDFs have no text layer; they are reported as an error
rather than silently indexed as empty (OCR is out of scope).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from .chunker import Segment

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".pdf", ".docx"})
MAX_FILE_BYTES = 50 * 1024 * 1024


class LoaderError(Exception):
    """The file could not be read or has no extractable text."""


@dataclass
class Document:
    source: str
    title: str
    segments: list[Segment]
    sha256: str

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        m = re.match(r"^#\s+(\S.*?)\s*#*\s*$", line.strip())
        if m:
            return m.group(1)
    return fallback


def _read_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


# ---- HTML ---------------------------------------------------------------------------------

class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    _BLOCK = {"p", "div", "section", "article", "ul", "ol", "table", "pre", "blockquote", "tr", "br", "main"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif re.fullmatch(r"h[1-6]", tag) or tag in self._BLOCK or tag == "li":
            self.parts.append("\n\n" if tag != "li" else "\n")

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        if not self._skip:
            self.parts.append(data)


def _html_to_text(html: str) -> tuple[str, str | None]:
    p = _HTMLText()
    p.feed(html)
    p.close()
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    title = " ".join("".join(p.title_parts).split()) or None
    return text, title


# ---- PDF / DOCX ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> tuple[list[Segment], str | None]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError as exc:  # pragma: no cover
        raise LoaderError("Reading PDFs needs pypdf: pip install pypdf") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise LoaderError("The PDF is password-protected")
        segments = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "")
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # undo end-of-line hyphenation
            if text.strip():
                segments.append(Segment(text, number))
        meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    except LoaderError:
        raise
    except (PyPdfError, OSError, ValueError, KeyError, TypeError) as exc:
        raise LoaderError(f"Could not read the PDF: {exc}") from exc
    if not segments:
        raise LoaderError("The PDF has no extractable text (scanned images? OCR is not supported)")
    return segments, meta_title or None


def _load_docx(path: Path) -> tuple[list[Segment], str | None]:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise LoaderError("Reading .docx needs python-docx: pip install python-docx") from exc
    try:
        d = docx.Document(str(path))
    except Exception as exc:  # python-docx raises assorted zip/xml errors
        raise LoaderError(f"Could not read the .docx file: {exc}") from exc

    lines: list[str] = []

    def paragraph(p) -> None:
        text = p.text.strip()
        if not text:
            lines.append("")
            return
        style = (p.style.name or "") if p.style is not None else ""
        m = re.match(r"Heading (\d)", style)
        if m:
            lines.append("#" * int(m.group(1)) + " " + text)
        elif style == "Title":
            lines.append("# " + text)
        elif "List" in style:
            lines.append("- " + text)
        else:
            lines.append(text)
        lines.append("")  # blank line: paragraph boundary

    def table(t) -> None:
        for row in t.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                lines.append("- " + " | ".join(cells))
        lines.append("")

    if hasattr(d, "iter_inner_content"):
        for item in d.iter_inner_content():
            (table if item.__class__.__name__ == "Table" else paragraph)(item)
    else:  # pragma: no cover  (older python-docx)
        for p in d.paragraphs:
            paragraph(p)
    text = "\n".join(lines).strip()
    if not text:
        raise LoaderError("The document contains no text")
    title = (d.core_properties.title or "").strip() or None
    return [Segment(text)], title


# ---- entry point --------------------------------------------------------------------------

def load_document(path: str | Path, source: str | None = None) -> Document:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoaderError(f"Unsupported file type {ext or '(none)'!r}")
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise LoaderError(f"File is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
        raw = path.read_bytes()
    except OSError as exc:
        raise LoaderError(f"Could not read the file: {exc}") from exc

    sha = hashlib.sha256(raw).hexdigest()
    fallback_title = path.stem.replace("_", " ").replace("-", " ").strip().title() or path.name
    title: str | None = None

    if ext == ".pdf":
        segments, title = _load_pdf(path)
    elif ext == ".docx":
        segments, title = _load_docx(path)
    elif ext in (".html", ".htm"):
        text, title = _html_to_text(_read_text(raw))
        segments = [Segment(text)] if text.strip() else []
    else:
        text = _read_text(raw).replace("\x00", "")
        segments = [Segment(text)] if text.strip() else []
        if ext in (".md", ".markdown"):
            title = _title_from_markdown(text, "") or None

    if not segments or not any(s.text.strip() for s in segments):
        raise LoaderError("The file contains no text")
    return Document(source or path.name, title or fallback_title, segments, sha)


def find_documents(root: str | Path) -> list[tuple[Path, str]]:
    """Return (path, source-name) pairs for a file or a directory tree (sorted, hidden files skipped)."""
    root = Path(root)
    if root.is_file():
        return [(root, root.name)]
    if not root.is_dir():
        raise LoaderError(f"Path not found: {root}")
    found = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not any(
            part.startswith(".") for part in rel.parts
        ):
            found.append((p, rel.as_posix()))
    return found
