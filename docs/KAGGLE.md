# Optional Kaggle batch evaluation

All application phases run locally. Kaggle is optional for sequential model comparison
on a larger evaluation subset. Enable Internet and a free GPU in notebook settings.
Do not upload copyrighted or sensitive PDFs. Use an external working directory and
keep results for download; Kaggle filesystem state is temporary.

Cell 1 (clone the completed project):

```python
!git clone https://github.com/GaurikMehrotra/-Research-Paper-Assistant.git /kaggle/working/research-paper-assistant
%cd /kaggle/working/research-paper-assistant
!python -m pip install -r requirements.txt
```

Kaggle supplies accelerator-enabled Torch. Do not replace it with the CPU wheel from
the local setup script. Check its version and hardware explicitly:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

Cell 2 (prepare open benchmark data and a fixed saved config):

```python
import yaml
from pathlib import Path
config = yaml.safe_load(Path("configs/default.yaml").read_text())
config["evaluation"]["subset"] = 30
config["runtime"]["device"] = "auto"
config["runtime"]["batch_size"] = 2
Path("configs/kaggle.yaml").write_text(yaml.safe_dump(config))
!python -m scripts.prepare_eval --questions 30 --summaries 5 --output data/eval/corpus.json
```

Cell 3 (sequential experiments and report):

```python
!python -m scripts.ablations --config configs/kaggle.yaml --corpus data/eval/corpus.json --output data/eval/runs/kaggle --compare-summaries
!python -m scripts.report data/eval/runs/kaggle
```

Cell 4 (download actual results):

```python
import shutil
from IPython.display import FileLink, display
archive = shutil.make_archive("/kaggle/working/rpa-results", "zip", "data/eval/runs/kaggle")
display(FileLink(archive))
```

Inspect every command's outcome. A failed dataset load or benchmark is not a completed
run. GPU latency cannot be reported as your i3 CPU latency. Sampled host RSS does not
include GPU VRAM. GPU execution and these cells have not been validated in this cloud
instance; dependency compatibility with Kaggle's preinstalled packages may require
an isolated kernel. No fine-tuning is required or implemented.
