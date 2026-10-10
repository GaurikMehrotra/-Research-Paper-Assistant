"""Reproducible benchmark runner for prepared, provenance-labelled corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil
import yaml

from src.eval.qa_eval import exact_f1
from src.eval.retrieval_eval import retrieval_metrics
from src.eval.summ_eval import summary_metrics
from src.generate.qa import answer_question
from src.generate.summarizer import summarize
from src.ingest.chunker import Chunk, chunk_passages
from src.ingest.pdf_parser import Passage
from src.pipeline import ResearchAssistant
from src.retrieval.embedder import fit_chunks
from src.retrieval.index import DenseIndex
from src.utils import fingerprint, write_json


@contextmanager
def memory_monitor(interval: float):
    """Sample process RSS including already loaded libraries; not laptop RAM guarantees."""
    stop = threading.Event()
    process = psutil.Process()
    measurements = [process.memory_info().rss]

    def sample() -> None:
        while not stop.wait(interval):
            measurements.append(process.memory_info().rss)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield measurements
    finally:
        measurements.append(process.memory_info().rss)
        stop.set()
        worker.join()


def prepare_papers(assistant: ResearchAssistant, corpus: dict) -> dict[str, list[Chunk]]:
    """Index normalized dataset paragraphs with dataset source-location provenance.

    These are dataset section/paragraph positions, not PDF pages. Dataset benchmarks
    bypass PDF parsing; reports must not claim they validate PDF layout extraction.
    """
    mapped = {}
    for paper in corpus["papers"]:
        pid = fingerprint([paper["id"], paper["passages"]])
        passages = [
            Passage(
                pid, paper["title"], p["page"], p["section"], p["text"], p.get("reference", False)
            )
            for p in paper["passages"]
        ]
        chunks = chunk_passages(passages, **assistant.config["chunking"])
        chunks = fit_chunks(
            chunks,
            assistant.config["models"]["embedding"],
            assistant.config["runtime"]["embedding_max_tokens"],
            assistant.config["runtime"]["embedding_overlap_tokens"],
            assistant.embedder.revision,
        )
        mapped[paper["id"]] = chunks
        directory = assistant.index_root / pid
        if not (directory / "metadata.json").exists():
            vectors = assistant.embedder.encode([c.text for c in chunks])
            DenseIndex(chunks, vectors, assistant.config["models"]["embedding"]).save(directory)
        write_json(
            assistant.root / f"{pid}.json",
            {"paper_id": pid, "paper": paper["title"], "chunks": [c.record() for c in chunks]},
        )
    return mapped


def run(config_path: str, corpus_path: str, output: str) -> dict:
    """Run retrieval, QA and summary measurements; persist raw predictions and resolved YAML."""
    corpus = json.loads(Path(corpus_path).read_text())
    assistant = ResearchAssistant(config_path)
    # Dataset paragraph ordinals are not PDF page numbers. Keep these artifacts
    # out of the application's upload cache so the UI can never serve them as PDFs.
    corpus_hash = hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest()
    assistant.root = assistant.root / "evaluation" / corpus_hash
    assistant.index_root = assistant.index_root / "evaluation" / corpus_hash
    assistant.root.mkdir(parents=True, exist_ok=True)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "config.yaml").write_text(yaml.safe_dump(assistant.config, sort_keys=True))
    report = {
        "configuration_sha256": fingerprint(assistant.config),
        "corpus_sha256": corpus_hash,
        "corpus": corpus.get("provenance", "unspecified"),
        "label_count": len(corpus["questions"]),
        "seed": assistant.config["seed"],
        "hardware": {
            "processor": __import__("platform").processor(),
            "platform": __import__("platform").platform(),
            "device": assistant.device,
            "threads": assistant.config["runtime"]["threads"],
            "logical_cpus": psutil.cpu_count(),
            "available_ram_mb": psutil.virtual_memory().total / 1024**2,
        },
    }
    rows = []
    with memory_monitor(assistant.config["evaluation"]["rss_sample_seconds"]) as rss:
        mapped = prepare_papers(assistant, corpus)
        rankings, relevance, ems, f1s, latencies = [], [], [], [], []
        subset = assistant.config["evaluation"]["subset"]
        for number, item in enumerate(corpus["questions"][:subset], 1):
            print(f"QA/retrieval {number}/{min(subset, len(corpus['questions']))}", file=sys.stderr)
            chunks = mapped[item["paper_id"]]
            pid = chunks[0].paper_id
            start = time.perf_counter()
            selected = assistant.retrieve(
                item["question"], pid, k=max(assistant.config["evaluation"]["k"])
            )
            latency = time.perf_counter() - start
            # Evidence may cover a sentence inside a longer dataset paragraph.
            gold = {
                c.id
                for c in chunks
                for text in item["evidence"]
                if text.strip() and (text in c.text or c.text in text)
            }
            if item["evidence"] and not gold:
                raise ValueError(
                    f"Evidence did not align to chunks: {item['id']}; inspect labels/chunks"
                )
            if gold:
                rankings.append([c.id for c, _ in selected])
                relevance.append(gold)
            start = time.perf_counter()
            result = answer_question(
                item["question"],
                selected[: assistant.config["retrieval"]["top_k"]],
                assistant.config,
                assistant.device,
            )
            qa_latency = time.perf_counter() - start
            prediction = "" if result["abstained"] else result["answer"]
            em, f1 = exact_f1(prediction, item["answers"])
            ems.append(em)
            f1s.append(f1)
            latencies.append(latency)
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "prediction": prediction,
                    "references": item["answers"],
                    "qa": result,
                    "em": em,
                    "f1": f1,
                    "retrieval_seconds": latency,
                    "qa_seconds": qa_latency,
                    "end_to_end_seconds": latency + qa_latency,
                    "faithful": "",
                    "citation_correct": "",
                    "review_notes": "",
                }
            )
        if not rows:
            raise ValueError("No questions executed; benchmark cannot report success")
        report["retrieval"] = retrieval_metrics(
            rankings, relevance, assistant.config["evaluation"]["k"]
        )
        report["retrieval"]["rank_depth"] = max(assistant.config["evaluation"]["k"])
        report["qa"] = {"em": sum(ems) / len(ems), "f1": sum(f1s) / len(f1s), "questions": len(ems)}
        report["mean_retrieval_seconds"] = sum(latencies) / len(latencies)
        report["mean_qa_seconds"] = sum(row["qa_seconds"] for row in rows) / len(rows)
        report["mean_query_seconds"] = sum(row["end_to_end_seconds"] for row in rows) / len(rows)
        write_json(
            output_path / "qa_checkpoint.json",
            {"status": "QA complete; summary pending", "report": report, "predictions": rows},
        )
        summary_rows = []
        summary_subset = assistant.config["evaluation"].get("summary_subset", subset)
        for number, item in enumerate(corpus.get("summaries", [])[:summary_subset], 1):
            print(
                f"Summary {number}/{min(summary_subset, len(corpus['summaries']))}", file=sys.stderr
            )
            # Convert every article paragraph to chunks; no article truncation.
            pid = fingerprint(item["article"])
            passages = [
                Passage(pid, item["id"], i + 1, "Dataset paragraph", text)
                for i, text in enumerate(item["article"].split("\n"))
                if text.strip()
            ]
            chunks = chunk_passages(passages, **assistant.config["chunking"])
            cache_key = fingerprint(
                [
                    assistant.config["models"]["summarizer"],
                    assistant.config.get("model_revisions", {}).get(
                        assistant.config["models"]["summarizer"]
                    ),
                    assistant.config["summary"],
                    assistant.config["seed"],
                    [c.record() for c in chunks],
                ]
            )
            cache_path = (
                Path(assistant.config["paths"]["processed"])
                / "eval-summaries"
                / f"{cache_key}.json"
            )
            if cache_path.exists():
                prediction = json.loads(cache_path.read_text())
            else:
                prediction = summarize(chunks, assistant.config, assistant.device)
                write_json(cache_path, prediction)
            summary_rows.append(
                {
                    "id": item["id"],
                    "prediction": prediction["summary"],
                    "reference": item["reference"],
                }
            )
        if summary_rows:
            report["summarization"] = summary_metrics(
                [r["prediction"] for r in summary_rows],
                [r["reference"] for r in summary_rows],
                assistant.config["evaluation"]["bertscore_model"],
                assistant.device,
                assistant.config["runtime"]["batch_size"],
                revision=assistant.config.get("model_revisions", {}).get(
                    assistant.config["evaluation"]["bertscore_model"]
                ),
                num_layers=assistant.config["evaluation"]["bertscore_layers"],
            )
        else:
            report["summarization"] = {"status": "not run: no summary samples selected"}
    report["peak_rss_mb"] = max(rss) / 1024**2
    write_json(output_path / "results.json", report)
    write_json(output_path / "predictions.json", rows)
    write_json(output_path / "summaries.json", summary_rows)
    with (output_path / "manual_review.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "prediction",
                "sources",
                "faithful",
                "citation_correct",
                "review_notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: row.get(k, "") for k in writer.fieldnames}
                | {"sources": json.dumps(row["qa"]["sources"])}
            )
    return report


def main() -> None:
    """Execute a benchmark from the terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fast_cpu.yaml")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.corpus, args.output), indent=2))


if __name__ == "__main__":
    main()
