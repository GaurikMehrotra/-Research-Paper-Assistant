# Final reproducibility checklist

- [x] Source code for ingestion, retrieval, QA, summaries, findings, notes, API and UI.
- [x] Fast CPU/default YAML, CPU install commands, no hardcoded `.cuda()`.
- [x] PDF and artifact exclusions; open-access-only arXiv helper.
- [x] Config-keyed processed/index/summary caches and exact provenance.
- [x] Software tests, lint, CPU CI definition, MIT license and model/system card.
- [x] Evaluation tools and measured results with sample limitations.
- [x] Hugging Face and arXiv network access enabled in the actual runtime.
- [x] Real pretrained-model ingest, cited QA, summarization and notes smoke validated.
- [x] QASPER retrieval and QA benchmarks actually executed.
- [x] arXiv summary metrics and FLAN-T5-base/DistilBART comparison executed.
- [ ] 30 user-written labels filled and evaluated; 20 real answers reviewed.
- [x] Ablation report and cloud CPU latency/RAM measured and documented.
- [ ] i3 laptop latency/RAM measured.
- [x] Model revisions and corpus hashes recorded for published quality results.
- [x] Real-model product demo recorded (an infrastructure demo is not model validation).
- [ ] GitHub CI outcome observed after pushing (a workflow file is not a passed CI run).

For the supplied existing repository, after local checks:

```bash
git status --short
git add .
git commit -m "Build CPU research paper assistant with cited retrieval and evaluation tools"
git push origin HEAD:main
```

Do not force-push. Recheck the remote if another author has created commits. Only for
a **new**, differently named repository, and after choosing your own name:

```bash
gh repo create research-paper-assistant --public --source=. --remote=origin --push
```

The existing supplied repository does not require `gh repo create`. Keep public/private
visibility unchanged when pushing. Configuration draft saving and environment
publication are separate from GitHub commits and pushes.
