"""Controlled one-factor ablations and sequential summarizer comparisons."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from src.eval.run import run
from src.utils import fingerprint, load_config


def main() -> None:
    """Keep the same seed, corpus and subset; isolate artifacts for every configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fast_cpu.yaml")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default="data/eval/runs/ablations")
    parser.add_argument("--compare-summaries", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = Path(args.config).resolve()
    base = yaml.safe_load(source.read_text())
    # Absolute paths prevent generated config directories from changing artifact locations.
    for key, path in base["paths"].items():
        base["paths"][key] = str((source.parent.parent / path).resolve())
    variants = {"baseline": base}
    edits = {
        "chunk-small": ("chunking", "words", 100),
        "chunk-large": ("chunking", "words", 220),
        "bge": ("models", "embedding", "BAAI/bge-small-en-v1.5"),
        "dense": ("retrieval", "mode", "dense"),
        "bm25": ("retrieval", "mode", "bm25"),
        "reranked": ("retrieval", "rerank", True),
    }
    if args.compare_summaries:
        edits.update(
            {
                "flan-base": ("models", "summarizer", "google/flan-t5-base"),
                "distilbart": ("models", "summarizer", "sshleifer/distilbart-cnn-12-6"),
            }
        )
    for name, (group, key, value) in edits.items():
        variant = copy.deepcopy(base)
        variant[group][key] = value
        if name not in {"flan-base", "distilbart"}:
            variant["evaluation"]["summary_subset"] = 0
        variants[name] = variant
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, config in variants.items():
        path = destination / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        result_path = destination / name / "results.json"
        previous = (
            json.loads(result_path.read_text()) if args.resume and result_path.exists() else {}
        )
        corpus_hash = hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()
        if (
            previous.get("configuration_sha256") == fingerprint(load_config(path))
            and previous.get("corpus_sha256") == corpus_hash
        ):
            reports[name] = previous
        else:
            print(f"Starting experiment: {name}", flush=True)
            reports[name] = run(str(path), args.corpus, str(destination / name))
        (destination / "comparison.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
