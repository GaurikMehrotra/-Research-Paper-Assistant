# Research Paper Assistant model and system card

This is a pretrained-model application, not a newly trained or independently
validated scientific model. Intended use: studying English research papers and
finding passages to verify. It is unsuitable for clinical decisions, legal advice,
automated literature conclusions, or treating extracted passages as verified facts.

| Task | Default | Fast CPU | Important limitation |
| --- | --- | --- | --- |
| Embeddings | BAAI/bge-small-en-v1.5 | sentence-transformers/all-MiniLM-L6-v2 | Semantic similarity is not entailment; English/domain bias |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Off unless enabled | MS MARCO relevance does not establish scientific support |
| Extractive QA | deepset/roberta-base-squad2 | Same | SQuAD2 training does not cover scientific reasoning or tables |
| Summaries | google/flan-t5-base | google/flan-t5-small | Can omit methods, invent claims and lose numerical detail |
| Comparison | sshleifer/distilbart-cnn-12-6 | google/flan-t5-base | CNN/news training is different from scientific writing |
| BERTScore encoder | distilbert-base-uncased | Same | Reference similarity is not faithfulness; score depends on encoder |

Upstream model cards, licenses, versions and dataset terms govern the weights and
data separately from this repository's MIT source license. Model commit revisions are pinned in YAML configs and recorded in
`data/eval/model_versions.json`. Dataset identity and corpus hashes are recorded with
published experiments. Local cache paths are infrastructure, not model identities.

No fine-tuning is performed. Device selection is automatic in the default config;
fast_cpu explicitly selects CPU. Model loads are serialized in each process and models
are released after use. This trades latency for memory. Do not run multiple workers or
both applications on an 8 GB laptop while measuring the memory target.

Support score = QA span score × nonnegative cosine similarity. This is an
**uncalibrated heuristic**, not a probability that an answer is true. Thresholds
are configuration defaults, not calibrated on QASPER. High-scoring incorrect spans
can pass. No-answer detection is imperfect. Abstention reports `not found in the paper`
with the retrieved passages; that phrase does not prove a paper contains no answer.

Summary citations identify map inputs, not verified support for every generated
sentence. Findings are keyword-selected source sentences, labelled candidates for
manual review. These design choices preserve inspectable evidence without claiming
that small language models can reliably establish scientific truth.

Measured baseline QASPER EM/F1 is 0.067/0.103 on 30 questions; Recall@5 is
0.300 on 29 evidence-labelled questions. BGE raises F1 to 0.158 on this tiny subset.
Summary ROUGE-L is 0.014/0.079/0.138 for FLAN-small/FLAN-base/DistilBART on two
arXiv examples. These source-ordered samples are not representative. See the saved
results and README for cloud CPU memory and latency measurements. A separate 30-question assistant-authored sample and 30 assistant source reviews are
included, with authorship disclosed. They are not independent human judgments; human
review is recommended. Blank user templates remain available and are not evaluation.

PDF extraction uses geometric and textual heuristics, without OCR or robust table,
equation, footnote, or arbitrary multi-column reconstruction. Scans and corrupt files
fail with an actionable message. Mixed scanned/text pages can lose content without
triggering the document-level minimum text check. PDF pages are one-based physical
positions, which can differ from printed page labels. QASPER does not provide PDF
pages; benchmark location fields explicitly describe paragraph ordinals.

All outputs may be wrong and must be verified against the original paper.
