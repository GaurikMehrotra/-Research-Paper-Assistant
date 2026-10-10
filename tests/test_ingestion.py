"""Parser and chunk boundary regression tests."""

import fitz
import pytest

from src.ingest.chunker import chunk_passages
from src.ingest.pdf_parser import Passage, PDFError, clean_text, parse_pdf
from src.ingest.sections import heading

PARSING = {
    "min_chars": 30,
    "margin_fraction": 0.07,
    "repeated_page_fraction": 0.6,
    "column_split_fraction": 0.5,
    "full_width_left_fraction": 0.45,
    "full_width_right_fraction": 0.55,
    "max_heading_words": 9,
}


def test_columns_headers_references(fixture_pdf):
    passages = parse_pdf(fixture_pdf, PARSING)
    assert passages[0].section == "Introduction"
    assert "Aurora" in passages[0].text
    assert "second column" in passages[1].text
    assert all("REPEATED" not in p.text for p in passages)
    assert passages[-1].reference and passages[-1].page == 2
    chunks = chunk_passages(passages, 30, 5)
    assert all(c.page == 1 and c.section == "Introduction" for c in chunks)


def test_hyphenation_and_heading():
    assert clean_text("re-\ntrieval with CPU-only models") == "retrieval with CPU-only models"
    assert heading("2.1 Evaluation", PARSING["max_heading_words"]) == "Evaluation"
    assert heading("This is a normal sentence.", PARSING["max_heading_words"]) is None
    assert heading("4 512", PARSING["max_heading_words"]) is None
    assert heading("5.77 6.11", PARSING["max_heading_words"]) is None


def test_scanned_and_corrupt(tmp_path):
    blank = tmp_path / "scan.pdf"
    with fitz.open() as document:
        document.new_page()
        document.save(blank)
    with pytest.raises(PDFError, match="OCR"):
        parse_pdf(blank, PARSING)
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a PDF")
    with pytest.raises(PDFError, match="corrupt"):
        parse_pdf(corrupt, PARSING)


def test_overlap_and_pages():
    passages = [
        Passage("id", "paper", 1, "Methods", "a b c d e f g"),
        Passage("id", "paper", 2, "Results", "h i j"),
    ]
    chunks = chunk_passages(passages, 4, 1)
    assert [c.text for c in chunks] == ["a b c d", "d e f g", "h i j"]
    assert [c.page for c in chunks] == [1, 1, 2]
    with pytest.raises(ValueError):
        chunk_passages(passages, 4, 4)
