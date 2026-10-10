PYTHON ?= python
CONFIG ?= configs/fast_cpu.yaml

.PHONY: install test lint format api ui papers eval ablations
install:
	$(PYTHON) -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
api:
	RPA_CONFIG=$(CONFIG) $(PYTHON) -m src.api
ui:
	RPA_CONFIG=$(CONFIG) $(PYTHON) -m app.app
papers:
	$(PYTHON) -m scripts.download_papers
eval:
	$(PYTHON) -m src.eval.run --config $(CONFIG) --corpus data/eval/corpus.json --output data/eval/runs/baseline
ablations:
	$(PYTHON) -m scripts.ablations --config $(CONFIG) --corpus data/eval/corpus.json --compare-summaries
