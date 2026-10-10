"""Offline end-to-end logic checks use deterministic injected model adapters."""

import numpy as np
import yaml
from fastapi.testclient import TestClient

from src.api import create_app
from src.eval.qa_eval import exact_f1
from src.eval.retrieval_eval import retrieval_metrics
from src.generate.findings import extract_findings
from src.generate.notes import render_notes
from src.generate.qa import answer_question
from src.generate.summarizer import token_windows
from src.pipeline import ResearchAssistant


def test_token_coverage_and_metrics():
    windows = token_windows(list(range(25)), 10, 2)
    assert set(n for window in windows for n in window) == set(range(25))
    assert exact_f1("the Aurora", ["Aurora"])[0] == 1
    metrics = retrieval_metrics([["b", "a"]], [{"a"}], [1, 2])
    assert metrics["recall@1"] == 0 and metrics["mrr"] == 0.5


def test_pipeline_and_api(fixture_pdf, tmp_path, monkeypatch):
    config = yaml.safe_load(open("configs/fast_cpu.yaml"))
    config["paths"] = {k: str(tmp_path / k) for k in config["paths"]}
    config["parsing"]["min_chars"] = 30
    config["retrieval"]["mode"] = "dense"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []

    def encode(self, texts, query=False):
        calls.append(query)
        return np.array([[1, 0] for _ in texts], dtype="float32")

    monkeypatch.setattr("src.retrieval.embedder.Embedder.encode", encode)
    monkeypatch.setattr("src.pipeline.fit_chunks", lambda chunks, *args: chunks)
    assistant = ResearchAssistant(path)
    paper = assistant.ingest(fixture_pdf)
    assert not paper["cached"]
    assert assistant.ingest(fixture_pdf)["cached"]
    assert calls == [False]  # No document model invocation on cache hit.
    evidence = assistant.retrieve("Which dataset?", paper["paper_id"])

    def predictor(**_):
        return {"answer": "Aurora", "score": 0.9}

    result = answer_question("Which dataset?", evidence, config, "cpu", predictor)
    assert result["answer"] == "Aurora"
    findings = extract_findings([c for c, _ in evidence])
    assert any(f["evidence"] for f in findings if f["field"] == "dataset")
    summary = {"summary": "Demo summary", "warning": "Verify", "map_summaries": []}
    assert "Aurora" in render_notes(paper["paper"], summary, findings)
    client = TestClient(create_app(str(path)))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/papers").json()[0]["paper_id"] == paper["paper_id"]
    assert client.post("/ask", json={"question": ""}).status_code == 422
    with fixture_pdf.open("rb") as file:
        response = client.post("/papers", files={"file": ("fixture.pdf", file, "application/pdf")})
    assert response.status_code == 200 and response.json()["cached"]
