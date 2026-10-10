"""Parse born-digital PDFs while retaining page and section provenance."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz

from src.ingest.sections import heading, is_references


class PDFError(ValueError):
    """A PDF is corrupt, encrypted, or needs OCR."""


@dataclass
class Passage:
    """One text block; pages are one-based PDF page positions, not printed labels."""

    paper_id: str
    paper: str
    page: int
    section: str
    text: str
    reference: bool = False


def clean_text(text: str) -> str:
    """Join line-wrap hyphenation and whitespace, preserving ordinary hyphenated words."""
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _ordered_blocks(page: fitz.Page, config: dict) -> list[tuple]:
    """Sort two-column bands around full-width blocks such as titles and figures."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    width = page.rect.width
    spanning = sorted(
        [
            b
            for b in blocks
            if b[0] < width * config["full_width_left_fraction"]
            and b[2] > width * config["full_width_right_fraction"]
        ],
        key=lambda b: b[1],
    )
    remaining = [b for b in blocks if b not in spanning]
    output = []
    for anchor in spanning:
        band = [b for b in remaining if b[1] < anchor[1]]
        output.extend(
            sorted(
                band, key=lambda b: (b[0] >= width * config["column_split_fraction"], b[1], b[0])
            )
        )
        remaining = [b for b in remaining if b not in band]
        output.append(anchor)
    output.extend(
        sorted(
            remaining, key=lambda b: (b[0] >= width * config["column_split_fraction"], b[1], b[0])
        )
    )
    return output


def parse_pdf(path: str | Path, config: dict) -> list[Passage]:
    """Remove repeated margin blocks and detect sections; raise actionable OCR errors."""
    path = Path(path)
    paper_id = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        document = fitz.open(path)
    except Exception as error:
        raise PDFError(f"Cannot open PDF {path.name}: corrupt or unsupported file") from error
    with document:
        if not document.is_pdf:
            raise PDFError("Expected a PDF; this file contains another document format")
        if document.is_encrypted:
            raise PDFError("Encrypted PDF: export an unencrypted copy first")
        pages = [_ordered_blocks(page, config) for page in document]
        if sum(len(clean_text(b[4])) for blocks in pages for b in blocks) < config["min_chars"]:
            raise PDFError("No usable text: scanned/empty PDF; run OCR before uploading")
        margins: Counter = Counter()
        fraction = config["margin_fraction"]
        for page, blocks in zip(document, pages, strict=True):
            candidates = {
                clean_text(b[4])
                for b in blocks
                if b[3] < page.rect.height * fraction or b[1] > page.rect.height * (1 - fraction)
            }
            margins.update(candidates)
        repeated = {
            text
            for text, count in margins.items()
            if count > 1 and count / len(pages) >= config["repeated_page_fraction"]
        }
        passages = []
        section = "Front matter"
        for number, blocks in enumerate(pages, 1):
            for block in blocks:
                text = clean_text(block[4])
                if not text or text in repeated or re.fullmatch(r"\d+", text):
                    continue
                # A heading may share a PDF block with the following paragraph.
                lines = block[4].strip().splitlines()
                heading_lines = 1
                title = lines[0]
                if len(lines) > 1 and re.fullmatch(r"\d+(?:\.\d+)*\.?", title.strip()):
                    title = title.strip() + " " + lines[1].strip()
                    heading_lines = 2
                detected = heading(title, config["max_heading_words"])
                if detected:
                    section = detected
                    text = clean_text("\n".join(lines[heading_lines:]))
                    if not text:
                        continue
                passages.append(
                    Passage(paper_id, path.name, number, section, text, is_references(section))
                )
        return passages


def to_records(passages: list[Passage]) -> list[dict]:
    """Serialize provenance without discarding references."""
    return [asdict(passage) for passage in passages]
