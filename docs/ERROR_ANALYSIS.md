# Error analysis protocol

Automatic baseline diagnostics: 18 of 30 answers abstained and seven had positive
reference token F1. These counts are not manual faithfulness judgments. FLAN-T5-small
produced an unrelated gamma-ray summary and repetitive “systyrofoam” text on the two
arXiv samples. Reranking improved Recall@5 slightly but reduced QA F1. Manual source
review is still required to assign error categories and validate generated claims.

1. Open the original PDF at the cited physical page. Compare extracted text and reading
   order with the page, including columns, captions, equations and references.
2. Check whether the cited passage actually answers the question, including negation,
   conditions, sample size, metric definitions and comparison baselines.
3. Inspect false abstentions: did retrieval fail, was a span below threshold, or was the
   question impossible for an extractive model (yes/no, multi-hop, arithmetic)?
4. Check summary claims against the map audit and original passages. Record invented
   claims, missing qualifiers, changed numbers, duplicated content and omitted results.
5. Check each findings candidate manually. Keyword matches often refer to related work,
   baselines or motivation rather than the paper's own result.

Fill at least 20 rows of the runner's `manual_review.csv`. Use `faithful=yes/no/unclear`,
`citation_correct=yes/no`, and descriptive `review_notes`. Do not judge faithfulness
from lexical overlap alone. Include easy, numerical, negated, multi-hop and absent
questions. Keep reviewers blind to retrieval variant when comparing ablations.

For each failed item record: question ID, expected answer, predicted span, paper,
section, page, exact evidence, error category, and a proposed diagnosis. Aggregate
counts only after review. An abstention on an answerable question is a QA failure,
not evidence of successful safety. Unsupported high-score answers are failures even
if token F1 is high. Incorrect citations are failures even if the answer text is right.

Use a development split for thresholds and a held-out test split for final reporting.
Report sample counts and label disagreement. Do not tune on your final test items or
claim hybrid/reranking helps based on one or two cherry-picked questions.
