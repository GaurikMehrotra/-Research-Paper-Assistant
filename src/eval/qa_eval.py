"""Standard normalized token EM/F1, including explicitly unanswerable questions."""

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    """Normalize articles, punctuation, case and spaces (SQuAD convention)."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_f1(prediction: str, references: list[str]) -> tuple[float, float]:
    """Use the best matching reference; empty answers represent unanswerable items."""
    if not references:
        raise ValueError("QA labels must include at least one answer (empty for unanswerable)")
    predicted = normalize(prediction)
    scores = []
    for reference in references:
        gold = normalize(reference)
        p, g = predicted.split(), gold.split()
        common = sum((Counter(p) & Counter(g)).values())
        f1 = float(p == g) if not p or not g else 2 * common / (len(p) + len(g))
        scores.append((float(predicted == gold), f1))
    return max(em for em, _ in scores), max(f1 for _, f1 in scores)
