"""Section/page bounded overlapping word chunks with immutable source identifiers."""

from dataclasses import asdict, dataclass

from src.ingest.pdf_parser import Passage
from src.utils import fingerprint


@dataclass
class Chunk:
    """A retrievable passage with exact provenance."""

    id: str
    paper_id: str
    paper: str
    page: int
    section: str
    text: str

    def record(self) -> dict:
        """Return a JSON record."""
        return asdict(self)


def chunk_passages(passages: list[Passage], words: int, overlap: int) -> list[Chunk]:
    """Combine adjacent blocks, never cross pages/sections, and omit reference sections."""
    if not 0 <= overlap < words:
        raise ValueError("Require 0 <= overlap < words")
    groups: list[tuple[Passage, list[str]]] = []
    for passage in passages:
        if passage.reference:
            continue
        if groups and (groups[-1][0].paper_id, groups[-1][0].page, groups[-1][0].section) == (
            passage.paper_id,
            passage.page,
            passage.section,
        ):
            groups[-1][1].extend(passage.text.split())
        else:
            groups.append((passage, passage.text.split()))
    chunks = []
    for passage, tokens in groups:
        for start in range(0, len(tokens), words - overlap):
            text = " ".join(tokens[start : start + words])
            identifier = fingerprint([passage.paper_id, passage.page, passage.section, start, text])
            chunks.append(
                Chunk(
                    identifier, passage.paper_id, passage.paper, passage.page, passage.section, text
                )
            )
            if start + words >= len(tokens):
                break
    return chunks
