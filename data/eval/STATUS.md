# Measurement status

Automated CPU evaluation completed: 30 QASPER questions, 29 evidence-labelled retrieval
items, two arXiv summaries, and nine retrieval/model/chunk variants. See
[results](results/RESULTS.md), metric JSON and resolved experiment YAMLs.

Models are revision-pinned. Corpus SHA256 and dataset provenance appear in each result.
QASPER source locations are paragraph ordinals, never PDF pages. Benchmark indexes are
isolated from uploaded PDF indexes. Samples are deterministic and source ordered;
these small comparisons do not establish general accuracy or statistical significance.

Assistant-authored PDF labels and source reviews now cover all 30 responses; see
`assistant_questions.csv`, `assistant_source_review.csv` and `results/assistant-authored`.
EM/F1 is 0.467/0.547. This convenience sample and its reviews are not independent
human evaluation. Human review and i3 measurements remain separate; templates are
provided for independently authored labels. No target-laptop RAM guarantee is claimed.

Cloud measurements use five logical CPUs, two Torch threads, approximately 33 GB RAM.
RSS is sampled every 50 ms and can miss brief peaks. Query timing includes model loads;
GPU RSS would exclude VRAM. Optional Kaggle instructions have not been executed.
