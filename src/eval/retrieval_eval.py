"""Question-level Recall@k and reciprocal rank on explicit relevant chunk labels."""


def retrieval_metrics(ranked: list[list[str]], relevant: list[set[str]], ks: list[int]) -> dict:
    """Macro recall counts all relevant chunks; MRR uses the first relevant result."""
    if not ranked or len(ranked) != len(relevant) or any(not gold for gold in relevant):
        raise ValueError("Require nonempty paired retrieval labels; exclude unanswerable items")
    result = {}
    for k in ks:
        result[f"recall@{k}"] = sum(
            len(set(ids[:k]) & gold) / len(gold) for ids, gold in zip(ranked, relevant, strict=True)
        ) / len(ranked)
    result["mrr"] = sum(
        next((1 / rank for rank, item in enumerate(ids, 1) if item in gold), 0)
        for ids, gold in zip(ranked, relevant, strict=True)
    ) / len(ranked)
    result["questions"] = len(ranked)
    return result
