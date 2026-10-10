"""Prepare bounded QASPER and arXiv/PubMed benchmark records without inventing labels."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download


def dataset_rows(repository: str, split: str, revision: str = "main") -> Iterator[dict]:
    """Cache official Parquet shards and read bounded batches without background scans.

    Remote Arrow streaming caused interpreter-shutdown races in the tested stack.
    Local Parquet batches with use_threads=False avoid that code path. Only the
    first needed shard is downloaded, not the full training corpus.
    """
    files = HfApi().list_repo_files(repository, repo_type="dataset", revision=revision)
    candidates = sorted(
        path
        for path in files
        if path.endswith(".parquet")
        and (path.startswith(f"qasper/{split}/") or path.startswith(f"document/{split}-"))
    )
    if not candidates:
        raise ValueError(f"No official Parquet shards found for {repository}/{split}")
    for filename in candidates:
        path = hf_hub_download(repository, filename, repo_type="dataset", revision=revision)
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=1, use_threads=False):
                yield from batch.to_pylist()


def qasper_records(count: int) -> tuple[list[dict], list[dict]]:
    """Retain extractive/free-form/yes-no and unanswerable annotations.

    QASPER lacks PDF page numbers. The page field here is a paragraph ordinal,
    explicitly identified in the section. This corpus evaluates retrieval/QA only.
    """
    dataset = dataset_rows("allenai/qasper", "validation", "refs/convert/parquet")
    papers, questions = [], []
    for row in dataset:
        passages = []
        number = 0
        for section, paragraphs in zip(
            row["full_text"]["section_name"], row["full_text"]["paragraphs"], strict=True
        ):
            for text in paragraphs:
                number += 1
                passages.append(
                    {
                        "page": number,
                        "section": f"{section} [dataset paragraph {number}]",
                        "text": " ".join(text.split()),
                    }
                )
        local = []
        for qid, question, annotations in zip(
            row["qas"]["question_id"], row["qas"]["question"], row["qas"]["answers"], strict=True
        ):
            answers, evidence = [], []
            for annotation in annotations["answer"]:
                if annotation["unanswerable"]:
                    answers.append("")
                elif annotation["extractive_spans"]:
                    answers.extend(annotation["extractive_spans"])
                elif annotation["free_form_answer"]:
                    answers.append(annotation["free_form_answer"])
                elif annotation["yes_no"] is not None:
                    answers.append("yes" if annotation["yes_no"] else "no")
                evidence.extend(
                    " ".join(text.split())
                    for text in annotation["evidence"]
                    if text.strip() and not text.startswith("FLOAT SELECTED")
                )
            if answers:
                local.append(
                    {
                        "id": qid,
                        "paper_id": row["id"],
                        "question": question,
                        "answers": answers,
                        "evidence": sorted(set(evidence)),
                    }
                )
        if local and passages:
            papers.append({"id": row["id"], "title": row["title"], "passages": passages})
            questions.extend(local)
        if len(questions) >= count:
            break
    dataset.close()
    return papers, questions[:count]


def main() -> None:
    """Save an auditable evaluation input; no executable remote dataset scripts are trusted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--summaries", type=int, default=2)
    parser.add_argument("--summary-dataset", choices=["arxiv", "pubmed"], default="arxiv")
    parser.add_argument("--output", default="data/eval/corpus.json")
    args = parser.parse_args()
    papers, questions = qasper_records(args.questions)
    summaries = []
    if args.summaries:
        dataset = dataset_rows(f"ccdv/{args.summary_dataset}-summarization", "test")
        for row in dataset:
            summaries.append(
                {
                    "id": row.get("article_id", str(len(summaries))),
                    "article": row["article"],
                    "reference": row["abstract"],
                }
            )
            if len(summaries) >= args.summaries:
                break
        dataset.close()
    questions = Dataset.from_list(questions).to_list()
    result = {
        "provenance": {
            "qa": "allenai/qasper validation",
            "summaries": f"ccdv/{args.summary_dataset}-summarization test",
            "location_kind": "dataset paragraph ordinal, NOT PDF page",
        },
        "papers": papers,
        "questions": questions,
        "summaries": summaries,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"Saved {len(questions)} questions and {len(summaries)} summaries to {path}")


if __name__ == "__main__":
    main()
