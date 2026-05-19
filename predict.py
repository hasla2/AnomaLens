import glob
import json
import os
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np

RISK_SCORE = {
    "noise": 0,
    "security_noise": 1,
    "important": 7,
    "error": 10,
    "anomaly": 15,  
}

OOD_THRESHOLD = 0.45  

model = joblib.load("model.joblib")


def find_files_to_process() -> list[str]:
    all_files = sorted(glob.glob("data/processed/events_*.jsonl"))
    if not all_files:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
    recent = []

    for path in all_files:
        basename = os.path.basename(path)
        ts_str = basename.replace("events_", "").replace(".jsonl", "")
        try:
            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
            if ts >= cutoff:
                recent.append(path)
        except ValueError:
            continue

    if recent:
        return recent

    print("No events files from the last 15 minutes — falling back to latest file")
    return [all_files[-1]]


files = find_files_to_process()

if not files:
    print("No events files found at all — exiting")
    exit(0)

for latest_file in files:
    basename = os.path.basename(latest_file)
    ts_part = basename.replace("events_", "").replace(".jsonl", "")
    out_path = f"data/processed/predictions_{ts_part}.jsonl"

    if os.path.exists(out_path):
        print(f"Skipping {basename} — already predicted")
        continue

    print(f"Processing: {basename}")

    with open(latest_file, encoding="utf-8") as f, open(
        out_path, "w", encoding="utf-8"
    ) as out:

        for line in f:
            event = json.loads(line)

            #  Classify with confidence (LinearSVC via decision_function)
            scores = model.decision_function([event["template"]])[0]
            best_idx = scores.argmax()
            pred = model.classes_[best_idx]

            # Softmax as confidence proxy
            exp_scores = np.exp(scores - scores.max())
            confidence = float(exp_scores[best_idx] / exp_scores.sum())

            # OOD: low confidence - flag as anomaly
            if confidence < OOD_THRESHOLD:
                pred = "anomaly"

            event["predicted_label"] = pred
            event["confidence"] = round(confidence, 3)
            event["risk_score"] = RISK_SCORE.get(pred, 0)

            out.write(json.dumps(event, ensure_ascii=False) + "\n")

            print(
                event["predicted_label"],
                event["risk_score"],
                round(confidence, 2),
                event.get("source", "unknown"),
                event["template"][:100],
            )

    print(f"Saved: {out_path}")
