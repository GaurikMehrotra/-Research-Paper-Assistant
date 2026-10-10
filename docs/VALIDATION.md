# Current-instance validation

- Python 3.12, CPU PyTorch 2.6.0, pip dependency consistency passed.
- Ruff lint/format passed; 12 offline pytest cases passed (upstream deprecation warnings).
- Actual Attention paper: repeat ingestion hit cache; architecture question answered
  “The Transformer” from Model Architecture, PDF page 3; unrelated capital question
  abstained. All 52 mapped windows were included in the summary audit; notes generated.
- FLAN-small summary was repetitive and unhelpful: “TFLOPS for the K80, K.” repeated.
  This failure is preserved in the demo rather than replaced with a curated summary.
- Nine CPU benchmark variants completed. Pinned BERTScore recomputation reproduced
  all three summarizer scores. See data/eval/results for metrics and experiment settings.
- Gradio and FastAPI startup/local requests were validated; final service checks
  are recorded below. No public deployment or fresh-snapshot restoration is claimed.
- Independent human review, i3 measurements and optional Kaggle remain unrun.

Final checks: Gradio HTTP root/config/API schema returned 200 and its ask endpoint
returned a highlighted Transformer answer. FastAPI cached upload, cited QA, summary,
findings and notes endpoints passed using the real models through TestClient.
