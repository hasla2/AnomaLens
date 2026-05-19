# FastAPI Interface

AnomaLens exposes a REST API for log classification, anomaly retrieval, and report access.

---

## Install FastAPI

```powershell
pip install fastapi uvicorn
```

---

## Run API

If `api.py` is located in the project `/src/api/api.py`:

```powershell
uvicorn src.api.api:app --host 0.0.0.0 --port 8000
```

Development mode with auto-reload:

```powershell
uvicorn src.api.api:app --reload
```

---

## Swagger UI

Open:

```text
http://localhost:8000/docs
```

FastAPI automatically generates interactive Swagger API documentation.

---

# API Endpoints

## 1. Classify Log Template

Classify a normalized log template using the trained ML model.

### Request

```text
GET /classify?template=<HOST> hostd failed to connect to vpxa
```

### Example

```text
http://localhost:8000/classify?template=<HOST> hostd failed to connect to vpxa
```

### Example Response

```json
{
  "label": "error",
  "confidence": 0.94,
  "risk_score": 10
}
```

---

## 2. Get Recent Anomalies

Return anomaly/error events detected during the specified time window.

### Request

```text
GET /anomalies?hours=12
```

### Example

```text
http://localhost:8000/anomalies?hours=12
```

---

## 3. Get Latest Report

Return the latest generated summary report.

### Request

```text
GET /report/latest
```

### Example

```text
http://localhost:8000/report/latest
```

---

# Usage from PowerShell

## Classify

```powershell
Invoke-RestMethod "http://localhost:8000/classify?template=hostd failed connect"
```

---

## Get anomalies for last 6 hours

```powershell
Invoke-RestMethod "http://localhost:8000/anomalies?hours=6"
```

---

## Get latest report

```powershell
Invoke-RestMethod "http://localhost:8000/report/latest"
```

---

# Usage from Python

## Classify

```python
import requests

r = requests.get(
    "http://localhost:8000/classify",
    params={"template": "<HOST> hostd failed to connect"}
)

print(r.json())
```

Example response:

```json
{
  "label": "error",
  "confidence": 0.94
}
```

---

## Get anomalies

```python
import requests

r = requests.get(
    "http://localhost:8000/anomalies",
    params={"hours": 12}
)

print(r.json())
```

---

# Architecture

```text
Graylog
   ↓
Collector
   ↓
Normalization
   ↓
ML Classification
   ↓
RAG / LLM Analysis
   ↓
FastAPI
   ↓
REST API / Slack Alerts
```

---

# Technologies

* FastAPI
* Uvicorn
* Scikit-learn
* Sentence Transformers
* Ollama
* Qdrant
* Slack SDK

---

# Notes

* The API runs locally by default on port `8000`
* Swagger documentation is available at `/docs`
* All timestamps are stored in UTC
* Anomaly detection uses confidence-based OOD logic
