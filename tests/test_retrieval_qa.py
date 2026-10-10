"""Index persistence, filtered retrieval, citations and conservative QA behavior."""

import numpy as np
import pytest

from src.generate.qa import answer_question, citation
from src.ingest.chunker import Chunk
from src.retrieval.bm25 import fuse, lexical_search
from src.retrieval.index import DenseIndex


def chunks():
    """Return a deterministic non-model corpus."""
    return [
        Chunk("a", "p1", "One.pdf", 1, "Results", "The Aurora dataset contains ten samples."),
        Chunk("b", "p2", "Two.pdf", 3, "Methods", "Neural retrieval uses vector embeddings."),
        Chunk("c", "p1", "One.pdf", 2, "Limitations", "English only; no clinical validation."),
    ]


def test_index_roundtrip_and_filter(tmp_path):
    index = DenseIndex(chunks(), np.array([[1, 0], [0, 1], [0.8, 0.2]], dtype="float32"), "fake")
    index.save(tmp_path)
    loaded = DenseIndex.load(tmp_path, "fake")
    assert loaded.search(np.array([1, 0]), 1)[0][0].id == "a"
    assert loaded.search(np.array([1, 0]), 3, "p2")[0][0].id == "b"
    assert loaded.search(np.array([1, 0]), 3, "missing") == []
    with pytest.raises(ValueError, match="mismatch"):
        DenseIndex.load(tmp_path, "other")


def test_hybrid():
    corpus = chunks()
    bm25 = lexical_search(corpus, "Aurora", 3)
    assert bm25[0][0].id == "a"
    fused = fuse([[(corpus[1], 0.9), (corpus[0], 0.1)], bm25], 2, 60)
    assert fused[0][0].id == "a"


def test_citation_and_abstention():
    config = {"qa": {"min_span_score": 0.3, "min_evidence_similarity": 0.2}}
    chunk = chunks()[0]

    def predictor(**_):
        return {"answer": "Aurora", "score": 0.9}

    result = answer_question("Which dataset?", [(chunk, 0.8)], config, "cpu", predictor)
    assert result["answer"] == "Aurora" and not result["abstained"]
    assert citation(chunk)["page"] == 1 and citation(chunk)["section"] == "Results"

    def weak(**_):
        return {"answer": "Aurora", "score": 0.01}

    assert answer_question("Unknown?", [(chunk, 0.8)], config, "cpu", weak)["abstained"]

    def invented(**_):
        return {"answer": "Invented", "score": 0.9}

    assert answer_question("Unknown?", [(chunk, 0.8)], config, "cpu", invented)["abstained"]
    result = answer_question("Unknown?", [], config, "cpu", predictor)
    assert result["answer"] == "not found in the paper" and result["sources"] == []
