"""Cached paper ingestion and serialized model orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from src.generate.findings import extract_findings
from src.generate.notes import render_notes
from src.generate.qa import answer_question
from src.generate.summarizer import summarize
from src.ingest.chunker import Chunk, chunk_passages
from src.ingest.pdf_parser import parse_pdf, to_records
from src.retrieval.bm25 import fuse, lexical_search
from src.retrieval.embedder import Embedder, fit_chunks
from src.retrieval.index import DenseIndex
from src.retrieval.reranker import rerank
from src.utils import configure_runtime, fingerprint, load_config, write_json

MODEL_LOCK = threading.RLock()


class ResearchAssistant:
    """All public calls serialize model loads; caches are keyed by PDF bytes and settings."""

    def __init__(self, config_path: str | Path) -> None:
        self.config = load_config(config_path)
        self.device = configure_runtime(self.config)
        self.embedder = Embedder(
            self.config["models"]["embedding"],
            self.device,
            self.config["runtime"]["batch_size"],
            self.config.get("model_revisions", {}).get(self.config["models"]["embedding"]),
        )
        self.namespace = fingerprint(
            {
                "parsing": self.config["parsing"],
                "parser_schema": 2,
                "chunking": self.config["chunking"],
                "embedding": self.config["models"]["embedding"],
                "embedding_revision": self.embedder.revision,
                "token_budget": self.config["runtime"]["embedding_max_tokens"],
                "token_overlap": self.config["runtime"]["embedding_overlap_tokens"],
            }
        )
        self.root = Path(self.config["paths"]["processed"]) / self.namespace
        self.index_root = Path(self.config["paths"]["indexes"]) / self.namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def _paper(self, paper_id: str) -> dict:
        if (
            not isinstance(paper_id, str)
            or len(paper_id) != 64
            or any(c not in "0123456789abcdef" for c in paper_id)
        ):
            raise ValueError("Invalid paper identifier")
        path = self.root / f"{paper_id}.json"
        if not path.exists():
            raise ValueError("Unknown paper; upload it with this configuration first")
        return json.loads(path.read_text())

    def papers(self) -> list[dict]:
        """List papers in this configuration's processed cache."""
        with MODEL_LOCK:
            return [
                {"paper_id": record["paper_id"], "paper": record["paper"]}
                for path in sorted(self.root.glob("*.json"))
                if "paper_id" in (record := json.loads(path.read_text()))
            ]

    def ingest(self, path: str | Path) -> dict:
        """Parse and embed once; subsequent identical uploads use disk artifacts."""
        with MODEL_LOCK:
            if Path(path).stat().st_size > self.config["server"]["max_upload_mb"] * 1024**2:
                raise ValueError("PDF exceeds the configured upload size limit")
            paper_id = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            cached = self.root / f"{paper_id}.json"
            directory = self.index_root / paper_id
            if cached.exists() and (directory / "metadata.json").exists():
                DenseIndex.load(directory, self.config["models"]["embedding"])
                return self._paper(paper_id) | {"cached": True}
            passages = parse_pdf(path, self.config["parsing"])
            chunks = chunk_passages(passages, **self.config["chunking"])
            chunks = fit_chunks(
                chunks,
                self.config["models"]["embedding"],
                self.config["runtime"]["embedding_max_tokens"],
                self.config["runtime"]["embedding_overlap_tokens"],
                self.embedder.revision,
            )
            if not chunks:
                raise ValueError("PDF contains no non-reference text")
            index = DenseIndex(
                chunks,
                self.embedder.encode([c.text for c in chunks]),
                self.config["models"]["embedding"],
            )
            index.save(directory)
            record = {
                "paper_id": paper_id,
                "paper": Path(path).name,
                "passages": to_records(passages),
                "chunks": [c.record() for c in chunks],
            }
            write_json(cached, record)
            return record | {"cached": False}

    def _index(self, paper_id: str | None = None) -> DenseIndex:
        ids = [paper_id] if paper_id else [p["paper_id"] for p in self.papers()]
        indexes = [
            DenseIndex.load(self.index_root / pid, self.config["models"]["embedding"])
            for pid in ids
            if self._paper(pid)
        ]
        if not indexes:
            raise ValueError("Upload at least one paper first")
        return DenseIndex(
            [c for i in indexes for c in i.chunks],
            np.concatenate([i.vectors for i in indexes]),
            self.config["models"]["embedding"],
        )

    def retrieve(
        self, question: str, paper_id: str | None = None, k: int | None = None
    ) -> list[tuple[Chunk, float]]:
        """Return selected candidates with cosine scores for QA's evidence threshold."""
        with MODEL_LOCK:
            if not question.strip():
                raise ValueError("Question must not be empty")
            index = self._index(paper_id)
            vector = self.embedder.encode([question], query=True)[0]
            settings = self.config["retrieval"]
            limit = settings["top_k"] if k is None else k
            candidate_count = max(settings["candidates"], limit)
            dense = index.search(vector, candidate_count, paper_id)
            if settings["mode"] == "hybrid":
                lexical = lexical_search(index.chunks, question, candidate_count, paper_id)
                selected = fuse([dense, lexical], candidate_count, settings["rrf_constant"])
            elif settings["mode"] == "bm25":
                selected = lexical_search(index.chunks, question, candidate_count, paper_id)
            elif settings["mode"] == "dense":
                selected = dense
            else:
                raise ValueError("Retrieval mode must be dense, hybrid or bm25")
            if settings["rerank"]:
                selected = rerank(
                    question,
                    selected,
                    self.config["models"]["reranker"],
                    self.device,
                    self.config["runtime"]["batch_size"],
                    self.config.get("model_revisions", {}).get(self.config["models"]["reranker"]),
                )
            positions = {c.id: n for n, c in enumerate(index.chunks)}
            return [
                (c, float(index.vectors[positions[c.id]] @ vector)) for c, _ in selected[:limit]
            ]

    def ask(self, question: str, paper_id: str | None = None) -> dict:
        """Retrieve sources and answer, or explicitly abstain."""
        with MODEL_LOCK:
            return answer_question(
                question, self.retrieve(question, paper_id), self.config, self.device
            )

    def summarize(self, paper_id: str) -> dict:
        """Cache generation separately from ingestion, invalidated by model/summary settings."""
        with MODEL_LOCK:
            key = fingerprint(
                [
                    paper_id,
                    self.config["models"]["summarizer"],
                    self.config["summary"],
                    self.config.get("model_revisions", {}).get(self.config["models"]["summarizer"]),
                ]
            )
            path = self.root / "summaries" / f"{key}.json"
            if path.exists():
                return json.loads(path.read_text())
            chunks = [Chunk(**c) for c in self._paper(paper_id)["chunks"]]
            result = summarize(chunks, self.config, self.device)
            write_json(path, result)
            return result

    def findings(self, paper_id: str) -> list[dict]:
        """Extract source candidates, without a model load."""
        return extract_findings([Chunk(**c) for c in self._paper(paper_id)["chunks"]])

    def notes(self, paper_id: str) -> str:
        """Produce notes with summary source audit and structured evidence."""
        return render_notes(
            self._paper(paper_id)["paper"], self.summarize(paper_id), self.findings(paper_id)
        )


def main() -> None:
    """Terminal interface for ingestion, retrieval, generation and notes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fast_cpu.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("papers")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("pdf")
    for command in ("summary", "findings", "notes"):
        sub.add_parser(command).add_argument("paper_id")
    for command in ("ask", "search"):
        child = sub.add_parser(command)
        child.add_argument("question")
        child.add_argument("--paper-id")
    args = parser.parse_args()
    assistant = ResearchAssistant(args.config)
    if args.command == "ingest":
        record = assistant.ingest(args.pdf)
        result = {k: v for k, v in record.items() if k not in {"chunks", "passages"}}
    elif args.command == "papers":
        result = assistant.papers()
    elif args.command == "ask":
        result = assistant.ask(args.question, args.paper_id)
    elif args.command == "search":
        result = [
            {"chunk": c.record(), "cosine": s}
            for c, s in assistant.retrieve(args.question, args.paper_id)
        ]
    elif args.command == "summary":
        result = assistant.summarize(args.paper_id)
    elif args.command == "findings":
        result = assistant.findings(args.paper_id)
    else:
        print(assistant.notes(args.paper_id))
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
