"""Normalized embeddings, loaded only for the duration of a call."""

from __future__ import annotations

import numpy as np

from src.ingest.chunker import Chunk
from src.utils import fingerprint, release_memory


def fit_chunks(
    chunks: list[Chunk], model_name: str, max_tokens: int, overlap: int, revision: str | None = None
) -> list[Chunk]:
    """Split overlong chunks at tokenizer offsets, preserving literal PDF source substrings."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, revision=revision)
    budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= overlap < budget:
        raise ValueError("Invalid embedding token overlap")
    result = []
    for chunk in chunks:
        encoded = tokenizer(chunk.text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if len(offsets) <= budget:
            result.append(chunk)
            continue
        for start in range(0, len(offsets), budget - overlap):
            end = min(start + budget, len(offsets))
            text = chunk.text[offsets[start][0] : offsets[end - 1][1]]
            result.append(
                Chunk(
                    fingerprint([chunk.id, start, text]),
                    chunk.paper_id,
                    chunk.paper,
                    chunk.page,
                    chunk.section,
                    text,
                )
            )
            if end == len(offsets):
                break
    return result


class Embedder:
    """CPU-friendly Sentence Transformer adapter; normalized dot product is cosine similarity."""

    def __init__(
        self, model: str, device: str, batch_size: int, revision: str | None = None
    ) -> None:
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self.revision = revision

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        """Encode without silently truncating document chunks exceeding model context."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, device=self.device, revision=self.revision)
        try:
            if query and "bge-" in self.model_name:
                texts = [
                    "Represent this sentence for searching relevant passages: " + t for t in texts
                ]
            lengths = [len(model.tokenizer(t)["input_ids"]) for t in texts]
            if any(length > model.max_seq_length for length in lengths):
                raise ValueError(
                    "Embedding input exceeds context; reduce chunk size or question length"
                )
            return model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
        finally:
            del model
            release_memory()
