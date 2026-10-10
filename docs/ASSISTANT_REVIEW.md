# Assistant-authored questions and source review

The assistant wrote labels from inspected Attention and BERT PDF passages before
running predictions, then inspected the resulting answers and all returned source
passages. It reviewed all 30 responses, exceeding the requested 20. This is disclosed
assistant review, not an independent human study. No quality thresholds were tuned
on these questions. The original blank human templates remain available.

The convenience sample contains 19 answerable Attention questions, nine answerable
BERT questions and one easy capital-of-Madagascar absence question for each paper.
Most questions are simple fact extraction; there are no representative multi-hop or
reasoning items. Assistant EM/F1 is 0.467/0.547; Recall@5 is 0.786 on 28 labelled
answerable items. This set must not be interpreted as stronger accuracy than QASPER.

## Findings from direct source inspection

- Eleven answerable questions falsely abstained. Six of those failures had the answer
  in a returned source: architecture, normalization, decoder layers, positional
  functions, GLUE score and next-sentence prediction. Other misses omitted the
  necessary source. Existing thresholds were retained rather than fitted to this set.
- The machine-translation beam question returned “21 and alpha = 0.3” from the
  constituency-parsing experiment (page 10). The correct translation beam size is
  four (Machine Translation, page 8). An exact span and real citation can still answer
  the wrong question. This is a substantive supported-span/wrong-context failure.
- “bidirectional” is a supported but incomplete answer to the BERT architecture
  question; the passage specifies a multi-layer bidirectional Transformer encoder.
- Extra variable notation such as “N = 6” lowers token metrics despite being faithful.
- Two responses supplied the right numeric fact from another section rather than the
  explicitly requested abstract. This source-choice limitation is recorded per row.
- Both deliberately easy absence questions abstained. They do not establish robust
  abstention on subtle, numerical or near-topic unanswerable questions.

All returned citation locations were checked against the matching extracted PDF text,
section and physical page. Here `citation_correct=yes` means the source location is
accurate; it does not mean the passage answers the question. Eighteen responses were
judged supported, including two easy correct abstentions; eleven false abstentions
and one wrong-context answer were judged unsupported as responses to their questions.
One of the supported spans was incomplete. Counts are specific to this assistant
reviewed set and are not a calibrated faithfulness estimate or human agreement rate.

`data/eval/assistant_source_review.csv` records each decision and its reason. Excerpts
are concise; full passages regenerate through the benchmark. `review_summary.json`
records authorship, scope and category counts. The review checker requires
`--reviewer-kind assistant` and rejects this CSV as human review. Independent, preferably
blinded human review remains recommended before making scientific reliability claims.

## Reproducibility

Download both pinned arXiv paper versions, run `scripts.prepare_sample_questions`,
convert with `scripts.custom_corpus --label-author assistant`, then execute the
30-question evaluation config. Commands are in README. Config, metrics and compact
predictions are saved under `data/eval/results/assistant-authored`. No PDFs are tracked.

Initial GitHub Actions CPU checks passed on Python 3.10 and 3.12 for commit 85319e5:
https://github.com/GaurikMehrotra/-Research-Paper-Assistant/actions/runs/38061279597
