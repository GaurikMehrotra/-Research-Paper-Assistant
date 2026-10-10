"""Create a CSV, Markdown results table and plot from completed benchmark artifacts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Report only executed metrics; missing metrics remain explicitly unmeasured."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = Path(args.directory)
    rows = []
    for path in sorted(directory.glob("**/results.json")):
        record = json.loads(path.read_text())
        predictions = json.loads((path.parent / "predictions.json").read_text())
        query_seconds = sum(
            row["retrieval_seconds"] + row["qa_seconds"] for row in predictions
        ) / len(predictions)
        rows.append(
            {
                "run": str(path.parent.relative_to(directory)),
                "questions": record["qa"]["questions"],
                "recall@5": record["retrieval"].get("recall@5", "not measured"),
                "mrr": record["retrieval"]["mrr"],
                "qa_em": record["qa"]["em"],
                "qa_f1": record["qa"]["f1"],
                "rougeL": record["summarization"].get("rougeL", "not measured"),
                "bertscore_f1": record["summarization"].get("bertscore_f1", "not measured"),
                "retrieval_seconds": record["mean_retrieval_seconds"],
                "query_seconds": query_seconds,
                "peak_rss_mb": record["peak_rss_mb"],
            }
        )
    if not rows:
        raise ValueError("No completed results found")
    with (directory / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in columns
            )
            + " |"
        )
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n")
    plt.bar([r["run"] for r in rows], [r["qa_f1"] for r in rows])
    plt.ylabel("QA token F1 (reference similarity, not faithfulness)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(directory / "qa_f1.png")


if __name__ == "__main__":
    main()
