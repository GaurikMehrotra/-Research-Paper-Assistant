"""Conservative section heading detection; PDF typography is not semantic markup."""

import re

KNOWN = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "methods",
    "method",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "datasets",
    "evaluation",
}


def heading(text: str, max_words: int) -> str | None:
    """Return a normalized heading only for short known or numbered titles."""
    text = text.strip()
    clean = re.sub(r"^\d+(?:\.\d+)*[.\s]+", "", text).strip().lower()
    if clean in KNOWN:
        return clean.title()
    title = re.sub(r"^\d+(?:\.\d+)*[.\s]+", "", text).strip()
    if (
        re.match(r"^\d+(?:\.\d+)*[.\s]+\S", text)
        and len(text.split()) <= max_words
        and re.match(r"[A-Z]", title)
    ):
        return title
    return None


def is_references(section: str) -> bool:
    """Reference sections remain available for lookup, but never enter QA indexes."""
    return section.lower() in {"references", "bibliography"}
