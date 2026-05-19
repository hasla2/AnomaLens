import json
from pathlib import Path

import joblib
from fastapi import FastAPI

app = FastAPI(title="AnomaLens API")
model = joblib.load("model.joblib")


@app.get("/classify")
def classify(template: str):
    pred = model.predict([template])[0]
    return {"label": pred, "template": template}


@app.get("/anomalies")
def anomalies(hours: int = 12):
    # читает predictions_*.jsonl
    ...


@app.get("/report/latest")
def latest_report():
    reports = sorted(Path("data/reports").glob("*.md"))
    return {"content": reports[-1].read_text()}
