"""Persist exact cosine FAISS indexes with JSON metadata and schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.ingest.chunker import Chunk
from src.utils import write_json


class DenseIndex:
    """Search only eligible vectors when a paper filter is supplied."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> None:
        if vectors.ndim != 2 or len(chunks) != len(vectors) or not len(chunks):
            raise ValueError("Nonempty chunks and matching 2D vectors required")
        self.chunks = chunks
        self.model_name = model_name
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        faiss.normalize_L2(self.vectors)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    def search(
        self, query: np.ndarray, k: int, paper_id: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Return top k cosine neighbors with an optional exact paper restriction."""
        if k <= 0:
            raise ValueError("k must be positive")
        eligible = [
            i for i, c in enumerate(self.chunks) if paper_id is None or c.paper_id == paper_id
        ]
        if not eligible:
            return []
        query = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        if query.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query/index embedding dimensions differ")
        faiss.normalize_L2(query)
        if paper_id is None:
            scores, ids = self.index.search(query, min(k, len(eligible)))
            return [
                (self.chunks[i], float(score)) for i, score in zip(ids[0], scores[0], strict=True)
            ]
        scores = self.vectors[eligible] @ query[0]
        order = np.argsort(-scores)[:k]
        return [(self.chunks[eligible[i]], float(scores[i])) for i in order]

    def save(self, directory: Path) -> None:
        """Save portable metadata; completion marker is written last."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        write_json(
            directory / "metadata.json",
            {"schema": 1, "model": self.model_name, "chunks": [c.record() for c in self.chunks]},
        )

    @classmethod
    def load(cls, directory: Path, expected_model: str) -> DenseIndex:
        """Reload without invoking any model or recomputing document embeddings."""
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["schema"] != 1 or metadata["model"] != expected_model:
            raise ValueError("Index schema/model mismatch: rebuild this index")
        index = faiss.read_index(str(directory / "index.faiss"))
        chunks = [Chunk(**c) for c in metadata["chunks"]]
        vectors = index.reconstruct_n(0, index.ntotal)
        return cls(chunks, vectors, expected_model)
