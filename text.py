"""Small text utilities shared by the chunker, embedder, retriever and demo model."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from as is are was were be been being am
    do does did done have has had having i you he she it we they me him her us them my your his its our their
    this that these those there here what which who whom whose when where why how can could should would will
    shall may might must not no nor so than too very just also about into over under up down out off per via
    any all each other some such only own same both few more most many much get gets long tell please give show
    s t""".split()
)

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def words(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens ("company's" -> company, s)."""
    return [w for w in _TOKEN.findall(text.lower())]


def stem(word: str) -> str:
    """A deliberately tiny suffix stripper. Only needs to be *consistent*, not linguistically correct."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ss"):
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("es") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def content_terms(text: str) -> list[str]:
    """Stemmed, stop-word-free terms in order (duplicates kept)."""
    return [stem(w) for w in words(text) if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]


def fts_query(text: str, max_terms: int = 32) -> str | None:
    """Turn free text into a safe FTS5 OR-query: every term is double-quoted, so user input can
    never be parsed as FTS syntax (NEAR, AND, column filters, quotes...). None if nothing is left."""
    terms = [w for w in words(text) if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]
    if not terms:
        terms = [w for w in words(text)]
    seen: dict[str, None] = {}
    for t in terms:
        seen.setdefault(t, None)
    unique = list(seen)[:max_terms]
    if not unique:
        return None
    return " OR ".join(f'"{t}"' for t in unique)
