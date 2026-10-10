# Fourteen-phase terminal guide

All phases are LOCAL; Phase 10 can optionally use KAGGLE. You authorized one continuous build, so this guide records the learning sequence without stopping between phases. Full code lives in the listed source files; no omitted-code placeholders are used. Run commands from the repository root in the activated environment. A phase depending on blocked downloads is not validated until that prerequisite is restored.

## Phase 1: Environment and PDF ingestion [LOCAL]

Keep dependencies isolated and inspect extracted reading order before trusting evidence.

Full implementations: `src/ingest/pdf_parser.py; scripts/download_papers.py; notebooks/01_parsing_checks.ipynb`.

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m scripts.download_papers 1706.03762v7
python -m pytest tests/test_ingestion.py -q
```

Expected: Parser tests pass. PDF download requires arXiv access; scans fail with an OCR message.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add CPU setup and provenance-preserving PDF ingestion"
```

## Phase 2: Sections and chunking [LOCAL]

Boundaries preserve exact citation pages and avoid indexing references.

Full implementations: `src/ingest/sections.py; src/ingest/chunker.py`.

```bash
python -m pytest tests/test_ingestion.py -q
```

Expected: Overlap/page tests pass and reference blocks are absent from QA chunks.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add section-aware overlapping chunks"
```

## Phase 3: Embeddings and indexes [LOCAL]

Persist document vectors so repeat uploads do not recompute them.

Full implementations: `src/retrieval/embedder.py; src/retrieval/index.py`.

```bash
python -m src.pipeline ingest data/papers/1706.03762v7.pdf
python -m src.pipeline ingest data/papers/1706.03762v7.pdf
python -m pytest tests/test_retrieval_qa.py -q
```

Expected: First ingest cached=false; second cached=true. Save/load/filter tests pass. Model download is required.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add normalized FAISS indexes and reload caching"
```

## Phase 4: Hybrid retrieval and reranking [LOCAL]

BM25 recovers lexical matches; RRF combines ranks without mixing score scales. Measure improvements rather than assume them.

Full implementations: `src/retrieval/bm25.py; src/retrieval/reranker.py`.

```bash
python -m src.pipeline search "What architecture is proposed?"
python -m pytest tests/test_retrieval_qa.py -q
```

Expected: Ranked chunks retain paper, section and page. Enable retrieval.rerank in a copied YAML to compare.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add BM25 hybrid retrieval and optional cross-encoder"
```

## Phase 5: Cited QA [LOCAL]

Literal span extraction and thresholds support abstention, while citations let you inspect mistakes.

Full implementations: `src/generate/qa.py`.

```bash
python -m src.pipeline ask "What architecture is proposed?"
python -m pytest tests/test_retrieval_qa.py -q
```

Expected: Answer JSON includes sources and heuristic score, or not found in the paper.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add cited extractive QA and abstention"
```

## Phase 6: Hierarchical summaries [LOCAL]

Cover every source token window and reduce outputs instead of truncating long papers.

Full implementations: `src/generate/summarizer.py`.

```bash
python -m src.pipeline papers
read -r -p "Paste the returned paper_id: " PAPER_ID
python -m src.pipeline summary "$PAPER_ID"
python -m pytest tests/test_generation_eval.py -q
```

Expected: Summary JSON includes map source audit and reduction rounds. Comparison runs are in Phase 10.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add token-budgeted hierarchical summarization"
```

## Phase 7: Findings and notes [LOCAL]

Evidence-linked candidates are more honest than inventing structured scientific conclusions.

Full implementations: `src/generate/findings.py; src/generate/notes.py`.

```bash
read -r -p "Paste paper_id: " PAPER_ID
python -m src.pipeline findings "$PAPER_ID"
python -m src.pipeline notes "$PAPER_ID" > study-notes.md
```

Expected: Candidate sentences have citations; Markdown contains warnings and source audit.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add evidence-linked findings and study notes"
```

## Phase 8: Cached pipeline and backend [LOCAL]

Use one reusable pipeline for terminal, API and UI behavior.

Full implementations: `src/pipeline.py; src/api.py`.

```bash
make api
# In another terminal:
curl -fsS http://127.0.0.1:8000/health
curl -fsS -F "file=@data/papers/1706.03762v7.pdf" http://127.0.0.1:8000/papers
```

Expected: Health reports CPU; upload returns a paper ID. Repeated upload hits cache.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add cached orchestration and FastAPI endpoints"
```

## Phase 9: Gradio interface [LOCAL]

Highlight escaped source passages so users can verify the extracted span.

Full implementations: `app/app.py`.

```bash
make ui
```

Expected: Local UI offers upload, paper selection, Q&A, summary, finding candidates and notes download.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add Gradio paper study interface"
```

## Phase 10: Evaluation and ablations [LOCAL]

Saved labels, fixed seeds and identical subsets make comparisons auditable.

Full implementations: `src/eval/; scripts/prepare_eval.py; scripts/ablations.py; scripts/report.py`.

```bash
python -m scripts.prepare_eval --questions 30 --summaries 2
make eval
python -m scripts.ablations --corpus data/eval/corpus.json --compare-summaries
python -m scripts.report data/eval/runs
```

Expected: Actual completed runs create predictions, config, metrics, table and plot. Network denial means not measured. See KAGGLE.md for optional GPU cells.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add reproducible retrieval QA summary and ablation evaluations"
```

## Phase 11: Manual faithfulness and error analysis [LOCAL]

Similarity metrics cannot determine whether the answer is faithful to the paper.

Full implementations: `data/eval/custom_questions.csv; data/eval/manual_faithfulness_review.csv; docs/ERROR_ANALYSIS.md`.

```bash
python -m scripts.custom_corpus
# Set evaluation.subset >= 30 in a copied experiment config before this command:
python -m src.eval.run --config configs/default.yaml --corpus data/eval/custom_corpus.json --output data/eval/runs/custom
python -m scripts.check_review data/eval/runs/custom/manual_review.csv --minimum 20
```

Expected: Complete 30 user labels first and use the copied config in the run. Blank templates fail. Fill 20 real judgements; checker then confirms count.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Document manual review and scientific failure modes"
```

## Phase 12: Quality checks and CI [LOCAL]

Tests exercise failure behavior and cache reuse without costly model downloads in CI.

Full implementations: `tests/; .github/workflows/ci.yml; Makefile; LICENSE`.

```bash
make lint test
python -m pip check
```

Expected: Ruff and tests pass locally. Observe remote CI separately after push.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Add CPU tests lint CI and licensing"
```

## Phase 13: Documentation and system card [LOCAL]

State what is measured, what is pending, and what outputs cannot establish.

Full implementations: `README.md; MODEL_CARD.md; docs/; notebooks/02_eval_analysis.ipynb`.

```bash
python -m src.pipeline --help
python -m src.eval.run --help
git diff --check
```

Expected: Commands are documented; known limitations and unmeasured scores remain explicit. Real-model demo records actual cited answers and an imperfect summary.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Document architecture setup measurement status and model limitations"
```

## Phase 14: Final review and GitHub [LOCAL]

Check exclusions and evidence before publishing claims.

Full implementations: `docs/CHECKLIST.md`.

```bash
make lint test
git status --short
git ls-files "*.pdf"
git add .
git commit -m "Build CPU research paper assistant with cited retrieval and evaluation tools"
git push origin HEAD:main
```

Expected: No tracked PDFs. Push updates existing repository without force. Unresolved measurements stay unchecked.

Suggested commit after your own phase changes:

```bash
git add .
git commit -m "Complete reproducibility and publication checklist"
```
