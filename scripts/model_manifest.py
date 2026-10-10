"""Record exact downloaded Hugging Face revisions for an experiment report."""

import argparse
import json
from pathlib import Path

from huggingface_hub import scan_cache_dir


def main() -> None:
    """Inspect model cache metadata, without accessing authentication files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/eval/model_versions.json")
    args = parser.parse_args()
    cache = scan_cache_dir()
    records = [
        {
            "repository": repo.repo_id,
            "type": repo.repo_type,
            "revisions": sorted(revision.commit_hash for revision in repo.revisions),
            "references": {
                revision.commit_hash: sorted(revision.refs) for revision in repo.revisions
            },
        }
        for repo in sorted(cache.repos, key=lambda r: r.repo_id)
    ]
    if not records:
        raise SystemExit("No cached revisions: do not claim pinned model evaluation")
    Path(args.output).write_text(json.dumps(records, indent=2))
    print(f"Recorded {len(records)} cached repositories")


if __name__ == "__main__":
    main()
