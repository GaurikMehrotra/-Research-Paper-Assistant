"""Configuration, reproducibility, and atomic artifact helpers."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve artifact paths relative to the config's project root."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    for key, value in config["paths"].items():
        config["paths"][key] = str((path.parent.parent / value).resolve())
    words, overlap = config["chunking"]["words"], config["chunking"]["overlap"]
    if not 0 <= overlap < words:
        raise ValueError("Chunk overlap must be nonnegative and smaller than chunk size")
    return config


def fingerprint(value: Any) -> str:
    """Hash JSON-serializable settings for cache isolation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically write an artifact so interrupted runs do not create valid-looking caches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as file:
        json.dump(value, file, indent=2)
        temporary = file.name
    os.replace(temporary, path)


def configure_runtime(config: dict) -> str:
    """Seed supported RNGs and select a portable Torch device."""
    import numpy as np
    import torch

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["runtime"]["threads"])
    device = config["runtime"]["device"]
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def release_memory() -> None:
    """Collect released models and free unused accelerator cache."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
