# Measurement status

Automated CPU evaluation completed: 30 QASPER questions, 29 evidence-labelled retrieval
items, two arXiv summaries, and nine retrieval/model/chunk variants. See
[results](results/RESULTS.md), metric JSON and resolved experiment YAMLs.

Models are revision-pinned. Corpus SHA256 and dataset provenance appear in each result.
QASPER source locations are paragraph ordinals, never PDF pages. Benchmark indexes are
isolated from uploaded PDF indexes. Samples are deterministic and source ordered;
these small comparisons do not establish general accuracy or statistical significance.

Pending user work: write 30 PDF-specific questions in `custom_questions.csv`, review at
least 20 generated answers using the provided CSV, and measure latency/RSS on the i3.
No manual faithfulness percentages or target-laptop RAM guarantees are claimed.

Cloud measurements use five logical CPUs, two Torch threads, approximately 33 GB RAM.
RSS is sampled every 50 ms and can miss brief peaks. Query timing includes model loads;
GPU RSS would exclude VRAM. Optional Kaggle instructions have not been executed.
