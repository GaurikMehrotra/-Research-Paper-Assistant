"""Evidence-linked structured findings; no invented dataset names or numeric metrics."""

import re

from src.generate.qa import citation
from src.ingest.chunker import Chunk

FIELDS = {
    "problem": r"\b(problem|challenge|aim|objective|propose)\b",
    "method": r"\b(method|approach|architecture|algorithm|model)\b",
    "dataset": r"\b(dataset|corpus|benchmark|data set)\b",
    "metrics": r"\b(accuracy|f1|precision|recall|bleu|rouge|metric|score)\b",
    "results": r"\b(outperform|achieve|result|improve)\w*\b",
    "limitations": r"\b(limitation|limited|future work|drawback|fail)\w*\b",
}


def extract_findings(chunks: list[Chunk]) -> list[dict]:
    """Return candidate source sentences for review, explicitly labelled as heuristics."""
    findings = []
    for field, pattern in FIELDS.items():
        candidates = []
        seen = set()
        for chunk in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
                if re.search(pattern, sentence, re.I) and sentence not in seen:
                    candidates.append({"text": sentence, "source": citation(chunk)})
                    seen.add(sentence)
        findings.append(
            {
                "field": field,
                "status": "candidate passages; manual verification required"
                if candidates
                else "not found in the paper",
                "evidence": candidates,
            }
        )
    return findings
