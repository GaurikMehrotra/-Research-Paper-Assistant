"""Test model orchestration and benchmark artifacts with explicit offline adapters."""

import json

import numpy as np
import pytest
import torch
import yaml

from src.eval.run import run
from src.generate.summarizer import summarize, token_windows
from src.ingest.chunker import Chunk
from src.pipeline import ResearchAssistant


def test_map_reduce_covers_sources(monkeypatch):
    """Every source window is generated, with bounded inputs and multiple reduction rounds."""
    calls = []

    class Tokenizer:
        model_max_length = 100

        def encode(self, text, add_special_tokens=False, verbose=False):
            return list(range(len(text.split())))

        def prepare_for_model(self, ids, **kwargs):
            assert len(ids) <= 6
            calls.append(ids)
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids))}

        def decode(self, ids, **kwargs):
            return "short summary"

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return torch.tensor([[1, 2]])

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(
        "transformers.AutoModelForSeq2SeqLM.from_pretrained", lambda *a, **k: Model()
    )
    config = {
        "models": {"summarizer": "offline-test"},
        "summary": {
            "max_input_tokens": 8,
            "num_beams": 1,
            "length_penalty": 1.0,
            "max_new_tokens": 3,
            "min_new_tokens": 1,
            "overlap_tokens": 2,
            "max_reduce_rounds": 8,
        },
    }
    chunk = Chunk("c", "p", "fixture.pdf", 2, "Methods", " ".join(f"w{i}" for i in range(100)))
    result = summarize([chunk], config, "cpu")
    windows = token_windows(list(range(100)), 6, 2)
    assert calls[: len(windows)] == windows
    assert len(result["map_summaries"]) == len(windows)
    assert result["reduce_rounds"] > 1
    assert result["map_summaries"][0]["source"]["page"] == 2


def test_eval_artifacts(tmp_path, monkeypatch):
    """Run actual metric and serialization logic, without pretending these are model scores."""
    config = yaml.safe_load(open("configs/fast_cpu.yaml"))
    config["paths"] = {k: str(tmp_path / k) for k in config["paths"]}
    config["retrieval"].update({"mode": "dense", "rerank": False})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    corpus = {
        "provenance": "synthetic offline software test, NOT a model benchmark",
        "papers": [
            {
                "id": "p1",
                "title": "Synthetic paper",
                "passages": [
                    {"page": 1, "section": "Methods", "text": "The Aurora dataset has ten samples."}
                ],
            }
        ],
        "questions": [
            {
                "id": "q1",
                "paper_id": "p1",
                "question": "Which dataset?",
                "answers": ["Aurora"],
                "evidence": ["The Aurora dataset has ten samples."],
            }
        ],
        "summaries": [],
    }
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus))
    monkeypatch.setattr("src.eval.run.fit_chunks", lambda chunks, *args: chunks)
    monkeypatch.setattr(
        "src.retrieval.embedder.Embedder.encode",
        lambda self, texts, query=False: np.array([[1, 0] for _ in texts], dtype="float32"),
    )

    def answer(*args, **kwargs):
        return {"answer": "Aurora", "abstained": False, "sources": [], "support_score": 1}

    monkeypatch.setattr("src.eval.run.answer_question", answer)
    output = tmp_path / "run"
    result = run(str(config_path), str(corpus_path), str(output))
    assert result["qa"]["questions"] == 1 and result["qa"]["f1"] == 1
    assert result["retrieval"]["recall@1"] == 1
    assert result["summarization"]["status"].startswith("not run")
    assert ResearchAssistant(config_path).papers() == []  # Dataset ordinals never become PDF pages.
    for file in ["results.json", "predictions.json", "manual_review.csv", "config.yaml"]:
        assert (output / file).exists()


def test_invalid_token_budget():
    with pytest.raises(ValueError):
        token_windows(list(range(10)), 3, 3)
