import re

import pytest

from rag_docqa.chunker import Chunk, Segment, chunk_document, count_words, split_sentences
from rag_docqa.store import Store
from rag_docqa.text import content_terms, fts_query, stem, words


# ---- text ------------------------------------------------------------------------------------

def test_stemmer_is_consistent_across_inflections():
    assert stem("expense") == stem("expenses")
    assert stem("approve") == stem("approved") == stem("approves")
    assert stem("policy") == stem("policies")
    assert stem("access") == "access"  # double-s words are not de-pluralised
    assert stem("24") == "24" and stem("is") == "is"


def test_content_terms_drop_stopwords_and_keep_numbers():
    assert content_terms("What is the daily meal allowance for 2 people?") == [
        stem("daily"), stem("meal"), stem("allowance"), "2", stem("people")
    ]


def test_words_handles_apostrophes_and_symbols():
    assert words("The company's $60 limit") == ["the", "company", "s", "60", "limit"]


@pytest.mark.parametrize(
    "hostile",
    [
        'foo" OR "bar', "NEAR(a b)", "a AND NOT b", "col:secret", "prefix*", "(((", '"', "'; DROP TABLE chunks; --",
        "-negated", "^start", "a OR", "AND", "NOT", "{x}", "\\", "title:leave", "\x00nul",
    ],
)
def test_fts_query_never_produces_fts_syntax_errors(hostile, indexed):
    q = fts_query(hostile)
    if q is not None:
        assert all(part.startswith('"') and part.endswith('"') for part in q.split(" OR "))
    indexed.fts_search(hostile, 5)  # must not raise
    indexed.fts_search(hostile, 5, doc_ids=[1])


def test_fts_query_empty_and_stopword_only():
    assert fts_query("") is None
    assert fts_query("???") is None
    assert fts_query("what is the") is not None  # falls back to the raw words rather than returning nothing


# ---- chunker ---------------------------------------------------------------------------------

DOC = """# Handbook

Intro paragraph about the handbook. It has two sentences.

## Leave

Employees get 24 days of leave. Unused days may carry over. Requests go through the portal.

- first bullet point text
- second bullet point text

## Travel

Economy class is required for short flights. Receipts are required for expenses over $25.
"""


def test_sections_become_paths_and_never_mix():
    chunks = chunk_document([Segment(DOC)], max_words=50, overlap_words=5)
    sections = [c.section for c in chunks]
    assert "Handbook" in sections and "Handbook > Leave" in sections and "Handbook > Travel" in sections
    for c in chunks:
        if c.section == "Handbook > Travel":
            assert "24 days" not in c.text
        if c.section == "Handbook > Leave":
            assert "Economy" not in c.text


def test_sibling_heading_replaces_previous_sibling_in_path():
    chunks = chunk_document([Segment("# A\n\n## B\n\nbody b text here.\n\n## C\n\nbody c text here.\n")])
    assert [c.section for c in chunks] == ["A > B", "A > C"]


def test_bullets_keep_their_own_lines():
    chunks = chunk_document([Segment(DOC)])
    leave = next(c for c in chunks if c.section == "Handbook > Leave")
    assert "- first bullet point text\n- second bullet point text" in leave.text


@pytest.mark.parametrize("max_words", [10, 25, 60, 180])
def test_no_chunk_exceeds_max_words_and_all_words_are_covered(max_words):
    long_text = " ".join(f"Sentence number {i} has a handful of ordinary words in it." for i in range(80))
    giant = "word" + " ".join(str(i) for i in range(500))  # one 500-token "sentence" with no punctuation
    segments = [Segment(f"# T\n\n{long_text}\n\n{giant}\n")]
    chunks = chunk_document(segments, max_words=max_words, overlap_words=min(5, max_words - 1))
    assert chunks and all(count_words(c.text) <= max_words for c in chunks)
    covered = set(" ".join(c.text for c in chunks).split())
    assert set(long_text.split()) | set(giant.split()) <= covered
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_trailing_sentences():
    sentences = [f"Fact number {i} is stated here clearly." for i in range(30)]
    chunks = chunk_document([Segment(" ".join(sentences))], max_words=40, overlap_words=16)
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(prev.text)[-1]
        assert last_sentence in nxt.text  # the boundary sentence appears on both sides


def test_zero_overlap_means_no_repetition():
    sentences = [f"Fact number {i} is stated here clearly." for i in range(30)]
    chunks = chunk_document([Segment(" ".join(sentences))], max_words=40, overlap_words=0)
    joined = " ".join(c.text for c in chunks)
    assert all(joined.count(s) == 1 for s in sentences)


def test_final_chunk_is_never_pure_overlap():
    sentences = [f"Fact number {i} is stated here clearly." for i in range(12)]
    chunks = chunk_document([Segment(" ".join(sentences))], max_words=30, overlap_words=12)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.text != prev.text and not prev.text.endswith(nxt.text)


def test_pages_are_tracked():
    segments = [Segment("First page sentence one. First page sentence two.", 1), Segment("Second page text here.", 2)]
    chunks = chunk_document(segments, max_words=200)
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2  # one chunk spans both pages
    chunks = chunk_document(segments, max_words=10, overlap_words=0)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2


@pytest.mark.parametrize("text", ["", "   \n\n  ", "# Only a heading", "#\n\n##"])
def test_degenerate_input_yields_no_chunks(text):
    assert chunk_document([Segment(text)]) == []


def test_parameter_validation():
    with pytest.raises(ValueError):
        chunk_document([Segment("x")], max_words=5)
    with pytest.raises(ValueError):
        chunk_document([Segment("x")], max_words=50, overlap_words=50)
    with pytest.raises(ValueError):
        chunk_document([Segment("x")], max_words=50, overlap_words=-1)


def test_windows_line_endings_and_wrapped_lines():
    text = "# T\r\n\r\nA sentence that is wrapped\r\nacross two lines. Another one.\r\n"
    (chunk,) = chunk_document([Segment(text)])
    assert "wrapped across two lines." in chunk.text and "\r" not in chunk.text


def test_index_text_adds_title_and_section_once():
    c = Chunk(0, "body", section="Doc Title > Part")
    assert c.index_text("Doc Title") == "Doc Title > Part\nbody"
    assert Chunk(0, "body", section="Part").index_text("Doc Title") == "Doc Title > Part\nbody"
    assert Chunk(0, "body").index_text("Doc Title") == "Doc Title\nbody"


def test_sentence_splitter_keeps_closing_quotes_and_decimals():
    parts = split_sentences('He said "stop." Then left. The rate is 3.5 percent. Next (see below). Done')
    assert parts[0] == 'He said "stop."'
    assert any("3.5 percent." in p for p in parts)
    assert all(re.search(r"\S", p) for p in parts)
