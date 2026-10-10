"""Optional cross-encoder reranking, never interpreted as a probability of truth."""

from src.ingest.chunker import Chunk
from src.utils import release_memory


def rerank(
    question: str,
    candidates: list[tuple[Chunk, float]],
    model_name: str,
    device: str,
    batch_size: int,
    revision: str | None = None,
) -> list[tuple[Chunk, float]]:
    """Score query-passage pairs and promptly unload the model."""
    from sentence_transformers import CrossEncoder

    if not candidates:
        return []
    model = CrossEncoder(model_name, device=device, revision=revision)
    try:
        scores = model.predict([(question, c.text) for c, _ in candidates], batch_size=batch_size)
        return sorted(
            [(c, float(s)) for (c, _), s in zip(candidates, scores, strict=True)],
            key=lambda item: item[1],
            reverse=True,
        )
    finally:
        del model
        release_memory()
