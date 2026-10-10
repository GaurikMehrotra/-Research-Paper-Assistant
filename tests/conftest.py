"""Tiny synthetic PDF fixtures are generated in temp folders, never committed as papers."""

import fitz
import pytest


@pytest.fixture
def fixture_pdf(tmp_path):
    """Create two columns, repeated margins, section heading, and references."""
    path = tmp_path / "fixture.pdf"
    with fitz.open() as document:
        for number in range(2):
            page = document.new_page(width=600, height=800)
            page.insert_text((40, 25), "REPEATED JOURNAL HEADER")
            if number == 0:
                page.insert_text((40, 100), "1\nIntroduction")
                page.insert_textbox(
                    fitz.Rect(40, 130, 270, 350),
                    "The experiment uses the Aurora dataset. "
                    "The method achieves 92 percent accuracy. "
                    "The proposed algorithm improves retrieval.",
                )
                page.insert_textbox(
                    fitz.Rect(330, 130, 560, 350),
                    "The second column discusses limitations. "
                    "The model is limited to English text.",
                )
            else:
                page.insert_text((40, 100), "References")
                page.insert_text((40, 130), "Smith et al. An unrelated citation, 2020.")
            page.insert_text((40, 780), "REPEATED JOURNAL FOOTER")
        document.save(path)
    return path
