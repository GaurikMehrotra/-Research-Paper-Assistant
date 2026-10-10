"""Extractive answers with explicit evidence and conservative abstention."""

from collections.abc import Callable

from src.ingest.chunker import Chunk
from src.utils import release_memory


def citation(chunk: Chunk) -> dict:
    """Return the exact source passage with one-based PDF page number."""
    return {
        "chunk_id": chunk.id,
        "paper_id": chunk.paper_id,
        "paper": chunk.paper,
        "section": chunk.section,
        "page": chunk.page,
        "passage": chunk.text,
    }


def answer_question(
    question: str,
    evidence: list[tuple[Chunk, float]],
    config: dict,
    device: str,
    predictor: Callable | None = None,
) -> dict:
    """Select a supported extractive span; scores are heuristic, uncalibrated signals.

    Evidence scores supplied here must be cosine similarities, never RRF or reranker logits.
    Retrieved passages are returned even on abstention to make failure inspectable.
    """
    owned = predictor is None
    if owned and evidence:
        from transformers import pipeline

        predictor = pipeline(
            "question-answering",
            model=config["models"]["qa"],
            device=device,
            revision=config.get("model_revisions", {}).get(config["models"]["qa"]),
        )
    candidates = []
    try:
        for chunk, similarity in evidence:
            if similarity < config["qa"]["min_evidence_similarity"]:
                continue
            result = predictor(question=question, context=chunk.text, handle_impossible_answer=True)
            span = result["answer"].strip()
            if span and span in chunk.text and result["score"] >= config["qa"]["min_span_score"]:
                candidates.append((result["score"], similarity, span, chunk))
    finally:
        if owned:
            del predictor
            release_memory()
    sources = [citation(c) for c, _ in evidence]
    if not candidates:
        return {
            "answer": "not found in the paper",
            "support_score": 0.0,
            "abstained": True,
            "sources": sources,
            "score_kind": "uncalibrated heuristic",
        }
    score, similarity, span, chunk = max(candidates, key=lambda item: item[0] * max(0, item[1]))
    return {
        "answer": span,
        "support_score": float(score * max(0, similarity)),
        "span_score": float(score),
        "evidence_similarity": float(similarity),
        "abstained": False,
        "sources": [citation(chunk)],
        "score_kind": "uncalibrated heuristic; does not establish entailment",
    }
