"""Lexical retrieval and reciprocal rank fusion (RRF)."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingest.chunker import Chunk


def tokenize(text: str) -> list[str]:
    """Use identical tokenization for questions and documents."""
    return re.findall(r"\b\w+\b", text.lower())


def lexical_search(
    chunks: list[Chunk], question: str, k: int, paper_id: str | None = None
) -> list[tuple[Chunk, float]]:
    """Build BM25 on the eligible corpus; skip meaningless zero-score matches."""
    eligible = [c for c in chunks if paper_id is None or c.paper_id == paper_id]
    if not eligible:
        return []
    scores = BM25Okapi([tokenize(c.text) for c in eligible]).get_scores(tokenize(question))
    return [(eligible[i], float(scores[i])) for i in np.argsort(-scores)[:k] if scores[i] > 0]


def fuse(
    rankings: list[list[tuple[Chunk, float]]], k: int, constant: int
) -> list[tuple[Chunk, float]]:
    """Fuse ranks rather than mixing incompatible cosine and BM25 score scales."""
    scores: dict[str, float] = {}
    chunks = {}
    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, 1):
            chunks[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (constant + rank)
    return [(chunks[key], scores[key]) for key in sorted(scores, key=scores.get, reverse=True)[:k]]
