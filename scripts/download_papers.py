"""Download a small open-access arXiv sample; PDFs are always ignored by Git."""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def download(identifier: str, directory: Path) -> dict:
    """Restrict input to arXiv IDs and retain source/hash metadata for reproducibility."""
    if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", identifier):
        raise ValueError("Expected a modern arXiv identifier, e.g. 1706.03762v7")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{identifier}.pdf"
    url = f"https://arxiv.org/pdf/{identifier}"
    if not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "ResearchPaperAssistant/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
        if not content.startswith(b"%PDF"):
            raise ValueError("arXiv did not return a PDF; retry later")
        destination.write_bytes(content)
    return {
        "id": identifier,
        "url": url,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "rights": "Open-access reading copy. Check each paper's license before redistribution.",
    }


def main() -> None:
    """Fetch an explicitly selected sample rather than arbitrary copyright-protected URLs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", default=["1706.03762v7", "1810.04805v2"])
    parser.add_argument("--directory", default="data/papers")
    args = parser.parse_args()
    records = [download(identifier, Path(args.directory)) for identifier in args.ids]
    Path(args.directory, "manifest.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
