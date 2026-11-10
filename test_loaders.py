import docx
import pytest

from rag_docqa import loaders
from rag_docqa.chunker import chunk_document
from rag_docqa.loaders import LoaderError, find_documents, load_document

from conftest import make_pdf


def test_markdown_title_and_text(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text("# Leave Policy\n\nBody text.", encoding="utf-8")
    d = load_document(p)
    assert d.title == "Leave Policy" and d.source == "policy.md" and "Body text" in d.segments[0].text
    assert len(d.sha256) == 64


def test_title_falls_back_to_file_name(tmp_path):
    p = tmp_path / "travel_and-expense_rules.txt"
    p.write_text("No headings here at all.", encoding="utf-8")
    assert load_document(p).title == "Travel And Expense Rules"


def test_utf8_bom_and_utf16(tmp_path):
    (tmp_path / "a.txt").write_bytes("﻿Hello wörld".encode("utf-8"))
    (tmp_path / "b.txt").write_bytes("Hello wörld".encode("utf-16"))
    assert load_document(tmp_path / "a.txt").segments[0].text == "Hello wörld"
    assert load_document(tmp_path / "b.txt").segments[0].text == "Hello wörld"


def test_html_extraction_drops_scripts_and_maps_structure(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><title>My Page</title><style>.x{}</style></head><body>"
        "<script>alert('evil')</script><h1>Main</h1><p>First para &amp; more.</p>"
        "<h2>Sub</h2><ul><li>one</li><li>two</li></ul><noscript>nope</noscript></body></html>",
        encoding="utf-8",
    )
    d = load_document(p)
    text = d.segments[0].text
    assert d.title == "My Page"
    assert "alert" not in text and "nope" not in text and ".x{}" not in text
    assert "# Main" in text and "## Sub" in text and "First para & more." in text
    assert "- one" in text and "- two" in text
    sections = {c.section for c in chunk_document(d.segments)}
    assert "Main > Sub" in sections


def test_pdf_pages_are_separate_segments(tmp_path):
    p = tmp_path / "report.pdf"
    p.write_bytes(make_pdf(["Alpha page one text.", "Beta page two text."]))
    d = load_document(p)
    assert [s.page for s in d.segments] == [1, 2]
    assert "Alpha" in d.segments[0].text and "Beta" in d.segments[1].text
    chunks = chunk_document(d.segments, max_words=10, overlap_words=0)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2


def test_pdf_without_text_layer_is_an_error(tmp_path):
    p = tmp_path / "scan.pdf"
    p.write_bytes(make_pdf([""]))
    with pytest.raises(LoaderError, match="no extractable text"):
        load_document(p)


def test_corrupt_pdf_is_an_error(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(LoaderError):
        load_document(p)


def test_docx_headings_lists_and_tables(tmp_path):
    d = docx.Document()
    d.add_heading("Company Handbook", 1)
    d.add_paragraph("Welcome paragraph.")
    d.add_heading("Benefits", 2)
    d.add_paragraph("Gym membership", style="List Bullet")
    t = d.add_table(rows=2, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Plan", "Cost"
    t.rows[1].cells[0].text, t.rows[1].cells[1].text = "Basic", "$10"
    p = tmp_path / "handbook.docx"
    d.save(p)
    doc = load_document(p)
    text = doc.segments[0].text
    assert "# Company Handbook" in text and "## Benefits" in text and "- Gym membership" in text
    assert "- Plan | Cost" in text and "- Basic | $10" in text
    assert text.index("Gym membership") < text.index("Plan | Cost")  # document order preserved


def test_empty_and_corrupt_docx(tmp_path):
    docx.Document().save(tmp_path / "empty.docx")
    with pytest.raises(LoaderError, match="no text"):
        load_document(tmp_path / "empty.docx")
    (tmp_path / "bad.docx").write_bytes(b"not a zip")
    with pytest.raises(LoaderError):
        load_document(tmp_path / "bad.docx")


@pytest.mark.parametrize("name, content", [("e.txt", ""), ("w.md", "  \n \n"), ("h.html", "<html><script>x()</script></html>")])
def test_files_without_text_are_errors(tmp_path, name, content):
    (tmp_path / name).write_text(content, encoding="utf-8")
    with pytest.raises(LoaderError):
        load_document(tmp_path / name)


def test_unsupported_extension_and_missing_file(tmp_path):
    (tmp_path / "x.exe").write_bytes(b"MZ")
    with pytest.raises(LoaderError, match="Unsupported"):
        load_document(tmp_path / "x.exe")
    with pytest.raises(LoaderError):
        load_document(tmp_path / "missing.txt")


def test_oversized_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "MAX_FILE_BYTES", 10)
    (tmp_path / "big.txt").write_text("x" * 100)
    with pytest.raises(LoaderError, match="larger than"):
        load_document(tmp_path / "big.txt")


def test_hash_tracks_content(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("one")
    h1 = load_document(p).sha256
    assert load_document(p).sha256 == h1
    p.write_text("two")
    assert load_document(p).sha256 != h1


def test_find_documents_recurses_sorts_and_skips_hidden(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.html").write_text("<p>c</p>")
    (tmp_path / "sub" / "skip.exe").write_text("x")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "d.txt").write_text("d")
    found = find_documents(tmp_path)
    assert [s for _, s in found] == ["a.md", "b.txt", "sub/c.html"]
    assert find_documents(tmp_path / "a.md")[0][1] == "a.md"
    with pytest.raises(LoaderError):
        find_documents(tmp_path / "nope")
