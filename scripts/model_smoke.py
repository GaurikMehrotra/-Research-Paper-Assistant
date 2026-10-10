"""Validate a real cited QA and map-reduce workflow on Attention Is All You Need."""

import argparse
import time
from pathlib import Path

from src.eval.run import memory_monitor
from src.pipeline import ResearchAssistant
from src.utils import write_json


def main() -> None:
    """Assertions test actual model behavior, not merely successful model construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fast_cpu.yaml")
    parser.add_argument("--paper", default="data/papers/1706.03762v7.pdf")
    parser.add_argument("--output", default="data/eval/functional-smoke.json")
    args = parser.parse_args()
    assistant = ResearchAssistant(args.config)
    start = time.perf_counter()
    with memory_monitor(assistant.config["evaluation"]["rss_sample_seconds"]) as rss:
        paper = assistant.ingest(args.paper)
        assert assistant.ingest(args.paper)["cached"]
        question = "What model architecture do the authors propose?"
        answer = assistant.ask(question, paper["paper_id"])
        assert not answer["abstained"] and "transformer" in answer["answer"].lower(), answer
        assert answer["answer"] in answer["sources"][0]["passage"]
        assert answer["sources"][0]["page"] >= 1
        absent_question = "What is the capital of Madagascar?"
        absent = assistant.ask(absent_question, paper["paper_id"])
        assert absent["abstained"] and absent["answer"] == "not found in the paper", absent
        summary = assistant.summarize(paper["paper_id"])
        assert summary["summary"] and summary["map_summaries"]
        covered = {item["source"]["chunk_id"] for item in summary["map_summaries"]}
        assert covered == {chunk["id"] for chunk in paper["chunks"]}
        notes = assistant.notes(paper["paper_id"])
        assert "## Summary source audit" in notes and "> " in notes
    record = {
        "paper_id": paper["paper_id"],
        "cached": paper["cached"],
        "qa": {question: {"result": answer}, absent_question: {"result": absent}},
        "summary": summary,
        "notes_bytes": len(notes),
        "seconds": time.perf_counter() - start,
        "peak_rss_mb": max(rss) / 1024**2,
    }
    write_json(Path(args.output), record)
    print(
        {
            "answer": answer["answer"],
            "source": answer["sources"][0] | {"passage": "see artifact"},
            "abstention": absent["answer"],
            "summary": summary["summary"],
            "mapped_windows": len(summary["map_summaries"]),
            "peak_rss_mb": record["peak_rss_mb"],
        }
    )


if __name__ == "__main__":
    main()
