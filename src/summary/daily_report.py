import glob
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ollama

PROCESSED_DIR = "data/processed"
REPORTS_DIR = "data/reports"
MODEL = "mistral7b"


def load_events_last_12h() -> list[dict]:
    # cutoff = datetime.utcnow() - timedelta(hours=24)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
    events = []
    # for f in glob.glob(f"{PROCESSED_DIR}/events_*.jsonl"):
    for f in glob.glob(f"{PROCESSED_DIR}/predictions_*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                    ts = e.get("timestamp", "")
                    if ts and ts >= cutoff.strftime("%Y-%m-%dT%H:%M:%S"):
                        events.append(e)
                except Exception:
                    continue
    return events


def aggregate(events: list[dict]) -> dict:
    labels = Counter(
        e.get("predicted_label") or e.get("label") or "unknown" for e in events
    )
    platforms = Counter(e.get("platform", "unknown") for e in events)
    services = Counter(e.get("service", "unknown") for e in events)
    errors = [e for e in events if e.get("predicted_label") in ("error", "anomaly")]
    unique_errors = list({e["template"]: e for e in errors}.values())[:5]

    return {
        # "period":        datetime.utcnow().strftime("%Y-%m-%d"),
        "period": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_events": len(events),
        "label_counts": dict(labels),
        "platforms": dict(platforms),
        "top_services": dict(services.most_common(5)),
        #        "top_errors":    [{"template": e["template"][:150],
        #                           "host":     e.get("host", "unknown")}
        #                          for e in unique_errors],
        "top_errors": [
            {
                "host": e.get("source", e.get("host", "unknown")),  #  rpint real host
                "template": e["template"][:150],
                "message": e.get("message", "")[:100],  # original message
            }
            for e in list(unique_errors)[:10]
        ],
    }


PROMPT = """\
You are an infrastructure operations analyst. Write a concise daily report in Markdown.

Data for {period}:
- Total log events: {total_events}
- Events by label: {label_counts}
- Events by platform: {platforms}
- Top services: {top_services}
- Top errors/anomalies: {top_errors}

Write the report with these sections:
## Daily Infrastructure Summary — {period}
### Overall Health (one sentence)
### Critical Events (list errors/anomalies with counts)
### Notable Patterns
### Recommendations

Be concise and actionable. Use bullet points."""


def generate_summary(stats: dict) -> str:
    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(**stats)}],
            options={"temperature": 0.3, "num_predict": 400, "keep_alive": "10m"},
        )
        try:
            return resp["message"]["content"]
        except (TypeError, KeyError):
            return resp.message.content
    except Exception as e:
        # Fallback — statistical report without LLM
        return f"""## Daily Infrastructure Summary — {stats['period']}

### Overall Health
Total events processed: {stats['total_events']}

### Events by Label
{chr(10).join(f"- {k}: {v}" for k, v in stats['label_counts'].items())}

### Top Errors
{chr(10).join(f"- {e['template'][:100]}" for e in stats['top_errors'])}

### Platforms
{chr(10).join(f"- {k}: {v}" for k, v in stats['platforms'].items())}

*LLM unavailable: {str(e)[:50]}*"""


def run():
    print("Loading events from the last 12 hours...")
    events = load_events_last_12h()
    print(f"Events found: {len(events)}")

    if not events:
        # Demo mode — take the latest file in full
        # files = sorted(glob.glob(f"{PROCESSED_DIR}/events_*.jsonl"))
        files = sorted(glob.glob(f"{PROCESSED_DIR}/predictions_*.jsonl"))
        if files:
            print(f"Demo-mode: take {files[-1]}")
            with open(files[-1], encoding="utf-8") as f:
                events = [json.loads(l) for l in f if l.strip()]
            print(f"Events for demo: {len(events)}")

    stats = aggregate(events)
    print(f"\nStats: {stats['label_counts']}")
    print("Generating summary...")

    summary = generate_summary(stats)

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    report_path = f"{REPORTS_DIR}/daily_{stats['period']}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n{'='*60}")
    print(summary)
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    run()
