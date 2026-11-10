"""Structure-aware chunking.

Why not "split every N characters"? Cutting mid-sentence and across headings produces chunks
that embed badly and cite badly. This chunker:

* keeps markdown/Word headings as a *section path* ("Leave Policy > Annual leave") and never
  lets a chunk straddle two sections;
* packs whole sentences (bullets stay separate units) up to ``max_words``;
* repeats a few trailing sentences at the start of the next chunk (``overlap_words``) so an
  answer that spans a boundary is still retrievable;
* splits a single over-long sentence by words, so **no chunk ever exceeds max_words**;
* records first/last page for PDFs so answers can cite "p. 3-4".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*#*\s*$")
_EMPTY_HEADING = re.compile(r"^#{1,6}$")  # a stray "#" with no title is noise, not content
_BULLET = re.compile(r"^\s*(?:[-*•+]|\d{1,3}[.)])\s+")
_SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9$])")


@dataclass
class Segment:
    """A run of text from a loader: one PDF page, or a whole file (page=None)."""

    text: str
    page: int | None = None


@dataclass
class Chunk:
    ordinal: int
    text: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    def index_text(self, title: str) -> str:
        """Text that is embedded and keyword-indexed: the passage plus its document title and
        section, so "leave policy" can match a chunk whose body never repeats those words."""
        if not self.section:
            header = title
        elif self.section.startswith(title):
            header = self.section  # the document's H1 already is the title
        else:
            header = f"{title} > {self.section}"
        return f"{header}\n{self.text}"


@dataclass
class _Unit:
    text: str
    lead: str  # separator that precedes this unit when joined ("\n\n" paragraph, "\n" bullet, " " sentence)
    page: int | None
    words: int


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def count_words(text: str) -> int:
    return len(text.split())


class _Builder:
    def __init__(self, max_words: int, overlap_words: int):
        self.max_words = max_words
        self.overlap_words = overlap_words
        self.chunks: list[Chunk] = []
        self.cur: list[_Unit] = []
        self.cur_words = 0
        self.new_units = 0  # units added since the last emit (guards against emitting pure overlap)
        self.stack: list[tuple[int, str]] = []
        self.section: str | None = None

    # -- structure ------------------------------------------------------------------------
    def heading(self, level: int, title: str) -> None:
        self.emit(keep_overlap=False)
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))
        self.section = " > ".join(t for _, t in self.stack)

    def add_unit(self, unit: _Unit) -> None:
        if self.cur and self.cur_words + unit.words > self.max_words:
            self.emit(keep_overlap=True)
            if self.cur and self.cur_words + unit.words > self.max_words:
                self.cur, self.cur_words = [], 0  # overlap would push us over the limit: drop it
        self.cur.append(unit)
        self.cur_words += unit.words
        self.new_units += 1

    def emit(self, keep_overlap: bool) -> None:
        if self.cur and self.new_units > 0:
            text = self.cur[0].text + "".join(u.lead + u.text for u in self.cur[1:])
            pages = [u.page for u in self.cur if u.page is not None]
            self.chunks.append(
                Chunk(
                    ordinal=len(self.chunks),
                    text=text.strip(),
                    section=self.section,
                    page_start=pages[0] if pages else None,
                    page_end=pages[-1] if pages else None,
                )
            )
        carry: list[_Unit] = []
        if keep_overlap and self.overlap_words > 0:
            total = 0
            for u in reversed(self.cur):
                if total + u.words > self.overlap_words:
                    break
                carry.insert(0, u)
                total += u.words
        self.cur = carry
        self.cur_words = sum(u.words for u in carry)
        self.new_units = 0

    # -- text -----------------------------------------------------------------------------
    def text_block(self, raw_units: list[str], page: int | None) -> None:
        for i, raw in enumerate(raw_units):
            for j, sentence in enumerate(split_sentences(raw)):
                lead = (("\n\n" if i == 0 else "\n") if j == 0 else " ")
                self._add_sentence(sentence, lead, page)

    def _add_sentence(self, sentence: str, lead: str, page: int | None) -> None:
        tokens = sentence.split()
        if len(tokens) <= self.max_words:
            self.add_unit(_Unit(sentence, lead, page, len(tokens)))
            return
        for k in range(0, len(tokens), self.max_words):  # one enormous "sentence": split by words
            part = tokens[k : k + self.max_words]
            self.add_unit(_Unit(" ".join(part), lead if k == 0 else " ", page, len(part)))


def chunk_document(segments: list[Segment], max_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    if max_words < 10:
        raise ValueError("max_words must be at least 10")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and < max_words")
    b = _Builder(max_words, overlap_words)
    for seg in segments:
        for block in re.split(r"\n\s*\n", seg.text.replace("\r\n", "\n").replace("\r", "\n")):
            pending: list[str] = []
            for line in block.split("\n"):
                stripped = line.strip()
                if not stripped or _EMPTY_HEADING.match(stripped):
                    continue
                m = _HEADING.match(stripped)
                if m:
                    if pending:
                        b.text_block(pending, seg.page)
                        pending = []
                    b.heading(len(m.group(1)), m.group(2))
                elif _BULLET.match(line) or not pending:
                    pending.append(stripped)
                else:
                    pending[-1] += " " + stripped
            if pending:
                b.text_block(pending, seg.page)
    b.emit(keep_overlap=False)
    return [c for c in b.chunks if c.text]
