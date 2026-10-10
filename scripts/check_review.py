"""Check whether a real manual review contains at least 20 completed judgements."""

import argparse
import csv


def main() -> None:
    """Never treat blank review templates as completed scientific evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--minimum", type=int, default=20)
    args = parser.parse_args()
    with open(args.csv, newline="") as file:
        rows = list(csv.DictReader(file))
    completed = [
        r
        for r in rows
        if r.get("id", "").strip()
        and r.get("question", "").strip()
        and r.get("sources", r.get("source_passage", "")).strip() not in {"", "[]"}
        and r.get("faithful", r.get("faithful_yes_no_unclear", "")) in {"yes", "no", "unclear"}
        and r.get("citation_correct", r.get("citation_correct_yes_no", "")) in {"yes", "no"}
    ]
    if len({r["id"] for r in completed}) != len(completed):
        raise SystemExit("Duplicate review IDs: review distinct real answers")
    if len(completed) < args.minimum:
        raise SystemExit(f"Incomplete review: {len(completed)}/{args.minimum} required judgements")
    print(
        f"{len(completed)} completed manual judgements; inspect disagreements and error categories"
    )


if __name__ == "__main__":
    main()
