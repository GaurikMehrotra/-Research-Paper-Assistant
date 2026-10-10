"""Convert the user's 30 labelled PDF questions into the benchmark corpus schema."""

import argparse
import csv
import json
from pathlib import Path

from src.pipeline import ResearchAssistant


def main() -> None:
    """Reject blank templates, missing evidence, and evidence not present in the cited PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fast_cpu.yaml")
    parser.add_argument("--labels", default="data/eval/custom_questions.csv")
    parser.add_argument("--output", default="data/eval/custom_corpus.json")
    args = parser.parse_args()
    assistant = ResearchAssistant(args.config)
    papers, questions = {}, []
    with Path(args.labels).open(newline="") as file:
        for row in csv.DictReader(file):
            if not row["question"].strip():
                raise ValueError(f"Blank question {row['id']}; complete the label template first")
            record = assistant._paper(row["paper_id"])
            papers[row["paper_id"]] = {
                "id": row["paper_id"],
                "title": record["paper"],
                "passages": record["passages"],
            }
            unanswerable = row["unanswerable"].strip().lower() == "true"
            evidence = [] if unanswerable else [row["evidence_passage"].strip()]
            if not unanswerable and (
                not evidence[0]
                or not any(
                    evidence[0] in p["text"]
                    and p["page"] == int(row["pdf_page"])
                    and p["section"] == row["section"]
                    for p in record["passages"]
                )
            ):
                raise ValueError(f"Source evidence/page/section does not match PDF: {row['id']}")
            answers = [""] if unanswerable else json.loads(row["reference_answers_json"])
            if (
                not isinstance(answers, list)
                or not answers
                or not all(isinstance(a, str) for a in answers)
            ):
                raise ValueError("reference_answers_json must be a nonempty list of strings")
            questions.append(
                {
                    "id": row["id"],
                    "paper_id": row["paper_id"],
                    "question": row["question"],
                    "answers": answers,
                    "evidence": evidence,
                }
            )
    corpus = {
        "provenance": "User-labelled questions with true PDF page/section locations",
        "papers": list(papers.values()),
        "questions": questions,
        "summaries": [],
    }
    Path(args.output).write_text(json.dumps(corpus, indent=2))


if __name__ == "__main__":
    main()
