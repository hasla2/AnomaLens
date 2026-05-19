# AnomaLens

Infrastructure log classification and incident explanation for VMware ESXi and Hyper-V environments.

**NLP Project · Spring 2026** | [Report (PDF)](report.pdf) | [Dataset sample](data/labels/labels_sample.csv)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![CPU only](https://img.shields.io/badge/inference-CPU--only-orange.svg)]()

---

## Overview

AnomaLens is a local NLP pipeline for server infrastructure log analysis. It runs entirely on CPU without GPU or external API calls, suitable for environments where log data cannot leave the corporate network.

The system does three things:
- Classifies log templates into four categories every 15 minutes (noise / important / error / security_noise)
- Retrieves relevant vendor documentation and generates explanations for detected errors
- Produces a structured 12-hour summary of infrastructure events

---

## Results

### Infrastructure log classification — proprietary dataset (n=64,744 templates)

| Model | Accuracy | Macro F1 | F1 error | F1 sec_n |
|---|---|---|---|---|
| Naive Bayes (CNB) | 0.917 | 0.884 | 0.938 | 0.858 |
| Random Forest | 0.986 | 0.959 | 0.966 | 0.891 |
| LogReg | 0.991 | 0.975 | 0.973 | 0.939 |
| **LinearSVC** | **0.994** | **0.987** | **0.978** | 0.978 |
| BERT-base (fine-tuned) | 0.957 | 0.957 | 0.972 | **0.988** |

LinearSVC 5-fold CV: macro F1 = 0.983 ± 0.003, 95% CI [0.978, 0.988]

### HDFS public benchmark (n=575,061 sessions)

| Model | P | R | F1 |
|---|---|---|---|
| **Random Forest** | **0.997** | **0.999** | **0.998** |
| LinearSVC | 0.972 | 1.000 | 0.986 |
| DeepLog [Du et al., 2017] | 0.962 | 0.938 | 0.949 |
| LogBERT [Guo et al., 2021] | 0.983 | 0.979 | 0.981 |

---

## Dataset

64,744 log templates from virtualized infrastructure collected via Graylog REST API. All hostnames replaced with `<HOST>`, IPs with `<IP>`.

| Label | Count | Share |
|---|---|---|
| noise | 45,533 | 70.3% |
| important | 11,002 | 17.0% |
| error | 6,937 | 10.7% |
| security_noise | 1,272 | 2.0% |

Anonymized sample (200 rows): [`data/labels/labels_sample.csv`](data/labels/labels_sample.csv)

---

## Hardware

Tested on Intel Xeon Silver 4214 (12 cores), 24 GB RAM, Windows Server, no GPU.

LLM models run via Ollama with Q4_K_M quantization:

| Model | Size | Role |
|---|---|---|
| Phi-3.5-Mini | 2.4 GB | RAG explanation |
| Mistral-7B-Instruct | 4.4 GB | Daily summary |

Note: `llama.cpp` was evaluated but failed on this CPU (AVX-512 without AVX-512_VNNI). Ollama was used instead.

---

## Stack

| Component | Technology |
|---|---|
| Log collection | Graylog REST API |
| Template normalization | Custom regex |
| Classification | scikit-learn TF-IDF + LinearSVC |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| Vector store | Qdrant (local file mode) |
| LLM inference | Ollama (Phi-3.5-Mini, Mistral-7B) |
| Scheduler | APScheduler |
| REST API | FastAPI |
| Alerts | Slack API |

---

## Installation

```bash
git clone https://github.com/hasla2/AnomaLens
cd AnomaLens
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Ollama setup

Download from [ollama.com/download](https://ollama.com/download/windows) and start:

```powershell
ollama serve
```

**Option A — pull models:**
```powershell
ollama pull mistral7b
ollama pull phi35
```

**Option B — import from local GGUF** (offline/corporate):
```powershell
echo "FROM models/mistral-7b-instruct-v0.2.Q4_K_M.gguf" > Modelfile7b
echo "FROM models/Phi-3.5-mini-instruct-Q4_K_M.gguf" > Modelfile_phi
ollama create mistral7b -f Modelfile7b
ollama create phi35 -f Modelfile_phi
```

### Embedding model

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("models/all-MiniLM-L6-v2")
```

### Configuration

```powershell
cp config.yaml.example config.yaml
# set graylog.url and graylog.token
```

### First run

```powershell
python src/rag/indexer.py   # index vendor docs once
python train_model.py       # train classifiers
```

---

## Usage

```powershell
# Full pipeline (15-min collect + 12h summary)
$env:NO_PROXY = "localhost,127.0.0.1"
python scheduler.py

# Single cycle test
python scheduler.py --once

# Individual components
python src/collector/graylog_collect.py
python predict.py
python src/summary/daily_report.py
python src/rag/retriever.py
```

---

## Evaluation

```powershell
# Classical classifiers (~5 sec)
python evaluate.py

# With cross-validation
python evaluate.py --crossval

# Include BERT-base
python evaluate.py --logbert

# Include Mistral-7B zero-shot (requires Ollama)
python evaluate.py --mistral --mistral-limit 200

# HDFS public benchmark
python hdfs_benchmark.py --data data/hdfs/preprocessed
```

---

## REST API

```powershell
uvicorn src.api.api:app --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

```
GET /classify?template=<HOST> hostd failed to connect to vpxa
GET /anomalies?hours=12
GET /report/latest
```

```python
import requests
r = requests.get("http://localhost:8000/classify",
                 params={"template": "<HOST> hostd failed to connect"})
print(r.json())
```

---

## Slack Alerts

```powershell
$env:SLACK_BOT_TOKEN = "xoxb-your-token-here"
python notify.py
```

Routes to `#alerts-critical` (P1/P2) or `#alerts-daily` (P3/P4) based on Mistral-7B severity assessment.

---

## BERT-base Fine-tuning

Fine-tuning is done on Google Colab (T4 GPU, ~30 min). See [`LogBERT_fine_tuning.ipynb`](LogBERT_fine_tuning.ipynb).

```powershell
# After downloading logbert_anomalens_final.zip from Colab:
Expand-Archive logbert_anomalens_final.zip -DestinationPath models\logbert_anomalens_final
pip install transformers
python evaluate.py --logbert
```

---

## Repository Structure

```
AnomaLens/
├── train_model.py
├── predict.py
├── evaluate.py
├── hdfs_benchmark.py
├── notify.py
├── scheduler.py
├── model.joblib
├── src/
│   ├── collector/graylog_collect.py
│   ├── rag/indexer.py
│   ├── rag/retriever.py
│   ├── summary/daily_report.py
│   └── api/api.py
├── data/
│   ├── labels/labels_sample.csv
│   └── results/
└── models/
```

---

## Ollama

```powershell
ollama list   # registered models
ollama ps     # running models
ollama stop mistral7b
```

---

## Citation

```bibtex
@misc{anomalens2026,
  title  = {AnomaLens: Infrastructure Log Analysis with LLM-Assisted RAG-Based Incident Explanation},
  author = {Natalia Iakovenko},
  year   = {2026},
  url    = {https://github.com/hasla2/AnomaLens}
}
```
