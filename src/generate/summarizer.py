"""Hierarchical token-budgeted summarization, with no silent source truncation."""

from src.generate.qa import citation
from src.ingest.chunker import Chunk
from src.utils import release_memory


def token_windows(tokens: list[int], budget: int, overlap: int) -> list[list[int]]:
    """Cover all tokens with overlap; invalid budgets fail rather than lose source text."""
    if not 0 <= overlap < budget:
        raise ValueError("Invalid summary token budget/overlap")
    windows = []
    for start in range(0, len(tokens), budget - overlap):
        windows.append(tokens[start : start + budget])
        if start + budget >= len(tokens):
            break
    return windows


def summarize(chunks: list[Chunk], config: dict, device: str) -> dict:
    """Map every chunk, then recursively reduce; keep a citation-bearing map audit trail."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = config["models"]["summarizer"]
    revision = config.get("model_revisions", {}).get(name)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, revision=revision).to(device)
    model.eval()
    settings = config["summary"]
    prompt = "Summarize the research passage: " if "t5" in name else ""
    prefix = tokenizer.encode(prompt, add_special_tokens=False, verbose=False)
    budget = min(settings["max_input_tokens"], tokenizer.model_max_length) - len(prefix) - 2

    def generate(ids: list[int]) -> str:
        prepared = tokenizer.prepare_for_model(prefix + ids, return_tensors="pt", truncation=False)
        inputs = {
            k: v.unsqueeze(0).to(device) if v.ndim == 1 else v.to(device)
            for k, v in prepared.items()
            if k in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                num_beams=settings["num_beams"],
                early_stopping=settings["num_beams"] > 1,
                length_penalty=settings["length_penalty"],
                max_new_tokens=settings["max_new_tokens"],
                min_new_tokens=settings["min_new_tokens"],
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    try:
        mapped = []
        for chunk in chunks:
            tokens = tokenizer.encode(chunk.text, add_special_tokens=False, verbose=False)
            for window in token_windows(tokens, budget, settings["overlap_tokens"]):
                mapped.append({"summary": generate(window), "source": citation(chunk)})
        if not mapped:
            raise ValueError("No non-reference source text to summarize")
        text = "\n".join(item["summary"] for item in mapped)
        rounds = 0
        previous_length = None
        while len(tokenizer.encode(text, add_special_tokens=False, verbose=False)) > budget:
            tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            if rounds >= settings["max_reduce_rounds"] or previous_length == len(tokens):
                raise RuntimeError(
                    "Summary reduction did not converge; no truncated summary returned"
                )
            previous_length = len(tokens)
            text = "\n".join(generate(w) for w in token_windows(tokens, budget, 0))
            rounds += 1
        final = (
            generate(tokenizer.encode(text, add_special_tokens=False, verbose=False))
            if len(mapped) > 1
            else text
        )
        return {
            "summary": final,
            "map_summaries": mapped,
            "reduce_rounds": rounds,
            "model": name,
            "revision": revision or getattr(getattr(model, "config", None), "_commit_hash", None),
            "warning": "Generated claims may be wrong; citations indicate inputs, "
            "not verified entailment. Check source passages.",
        }
    finally:
        model = None
        tokenizer = None
        release_memory()
