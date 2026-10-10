"""Summarization metrics measure reference similarity, not factual correctness."""

from pathlib import Path

from src.utils import release_memory


def summary_metrics(
    predictions: list[str],
    references: list[str],
    model: str,
    device: str,
    batch_size: int,
    revision: str | None = None,
    num_layers: int | None = None,
) -> dict:
    """Compute ROUGE and BERTScore with a recorded encoder, unloading between stages."""
    import evaluate

    if not predictions or len(predictions) != len(references):
        raise ValueError("Require nonempty paired summaries and references")
    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    del rouge
    release_memory()
    scorer = evaluate.load("bertscore")
    model_path = model
    if revision:
        from huggingface_hub import snapshot_download, try_to_load_from_cache

        configuration = try_to_load_from_cache(model, "config.json", revision=revision)
        root = Path(configuration).parent if isinstance(configuration, str) else None
        if root and (
            (root / "model.safetensors").exists() or (root / "pytorch_model.bin").exists()
        ):
            model_path = str(root)
        else:
            model_path = snapshot_download(
                model,
                revision=revision,
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
            )
    scores = scorer.compute(
        predictions=predictions,
        references=references,
        model_type=model_path,
        num_layers=num_layers,
        device=device,
        batch_size=batch_size,
    )
    result.update(
        {
            "bertscore_f1": sum(scores["f1"]) / len(scores["f1"]),
            "bertscore_hash": scores["hashcode"],
            "bertscore_model": model,
            "bertscore_revision": revision,
            "bertscore_layers": num_layers,
            "samples": len(predictions),
        }
    )
    del scorer
    release_memory()
    return result
