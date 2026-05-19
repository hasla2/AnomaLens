"""
notify.py — LLM-based alert routing for AnomaLens.

Reads latest predictions, uses Mistral-7B to decide severity,
and routes to the appropriate Slack channel.

Channels:
    #alerts-critical  — P1/P2: immediate action required
    #alerts-daily     — P3/P4: informational, review when convenient
"""

import glob
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ollama
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

SLACK_TOKEN = os.environ["SLACK_BOT_TOKEN"]  # set in environment
CHANNEL_URGENT = os.environ["SLACK_CHANNEL_URGENT"]
CHANNEL_DAILY = os.environ["SLACK_CHANNEL_DAILY"]
MODEL = "mistral7b"
PROCESSED_DIR = Path("data/processed")

# ─── Load recent anomalies ────────────────────────────────────────────────────


def load_recent_errors(minutes: int = 15) -> list[dict]:
    """Load error/anomaly events from the last N minutes."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    events = []

    for path in sorted(PROCESSED_DIR.glob("predictions_*.jsonl")):
        ts_str = path.stem.replace("predictions_", "")
        try:
            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
            if ts < cutoff:
                continue
        except ValueError:
            continue

        with open(path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if e.get("predicted_label") in ("error", "anomaly"):
                    events.append(e)

    return events


# ─── LLM severity decision ────────────────────────────────────────────────────

SEVERITY_PROMPT = """\
You are an infrastructure on-call analyst.
Analyze these log anomalies and decide if they require IMMEDIATE action.

Anomalies detected (last 15 minutes):
{anomalies}

Reply with ONLY this JSON:
{{
  "urgent": true or false,
  "severity": "P1" or "P2" or "P3" or "P4",
  "reason": "one sentence explanation",
  "affected_hosts": ["host1", "host2"]
}}

P1 = service down, data loss risk
P2 = degraded performance, risk of escalation  
P3 = warning, monitor closely
P4 = informational, review later"""


def assess_severity(events: list[dict]) -> dict:
    """Ask LLM to assess if anomalies require urgent attention."""
    if not events:
        return {
            "urgent": False,
            "severity": "P4",
            "reason": "No anomalies detected",
            "affected_hosts": [],
        }

    # Format top-5 anomalies for LLM
    top = sorted(events, key=lambda x: x.get("risk_score", 0), reverse=True)[:5]
    anomaly_text = "\n".join(
        f"- [{e.get('source','unknown')}] {e['template'][:120]}" for e in top
    )

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": SEVERITY_PROMPT.format(anomalies=anomaly_text),
                }
            ],
            format="json",
            options={"temperature": 0.1, "num_predict": 200, "keep_alive": "10m"},
        )
        try:
            raw = resp["message"]["content"]
        except (TypeError, KeyError):
            raw = resp.message.content

        return json.loads(raw)

    except Exception as e:
        # Fallback: if LLM unavailable, use rule-based decision
        has_p1 = any(e.get("risk_score", 0) >= 15 for e in events)
        return {
            "urgent": has_p1,
            "severity": "P1" if has_p1 else "P3",
            "reason": f"LLM unavailable, rule-based: {len(events)} anomalies",
            "affected_hosts": list({e.get("source", "?") for e in events})[:3],
        }


# ─── Slack notification ───────────────────────────────────────────────────────


def send_slack(channel: str, blocks: list) -> bool:
    """Send a Slack message with Block Kit blocks."""
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={
            "Authorization": f"Bearer {SLACK_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"channel": channel, "blocks": blocks},
        timeout=10,
    )
    data = resp.json()
    if not data.get("ok"):
        print(f"Slack error: {data.get('error')}")
        return False
    return True


def build_urgent_message(assessment: dict, events: list[dict]) -> list:
    """Build Slack Block Kit message for urgent alerts."""
    hosts = ", ".join(assessment.get("affected_hosts", [])[:3]) or "unknown"
    severity = assessment.get("severity", "P?")
    reason = assessment.get("reason", "")
    count = len(events)

    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🚨 {severity} Infrastructure Alert",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Anomalies:*\n{count}"},
                {"type": "mrkdwn", "text": f"*Affected hosts:*\n{hosts}"},
                {"type": "mrkdwn", "text": f"*Assessment:*\n{reason}"},
                {
                    "type": "mrkdwn",
                    "text": f"*Time:*\n{datetime.now(timezone.utc).strftime('%H:%M UTC')}",
                },
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Top anomalies:*\n"
                + "\n".join(
                    f"• `{e.get('source','?')}` {e['template'][:80]}"
                    for e in sorted(
                        events, key=lambda x: x.get("risk_score", 0), reverse=True
                    )[:3]
                ),
            },
        },
    ]


def build_daily_message(assessment: dict, events: list[dict]) -> list:
    """Build Slack Block Kit message for daily/informational alerts."""
    count = len(events)
    reason = assessment.get("reason", "")

    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"ℹ️ *AnomaLens update* — "
                    f"{count} event(s) classified as error/anomaly. "
                    f"{reason}"
                ),
            },
        }
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────


def run():
    print("Loading recent errors...")
    events = load_recent_errors(minutes=15)
    print(f"  Found {len(events)} error/anomaly events")

    if not events:
        print("  Nothing to report.")
        return

    print("Assessing severity with LLM...")
    assessment = assess_severity(events)
    print(
        f"  Severity: {assessment['severity']} | "
        f"Urgent: {assessment['urgent']} | "
        f"Reason: {assessment['reason']}"
    )

    # Route to appropriate channel
    if assessment["urgent"]:
        channel = CHANNEL_URGENT
        blocks = build_urgent_message(assessment, events)
        print(f"  → Sending to #{channel} (URGENT)")
    else:
        channel = CHANNEL_DAILY
        blocks = build_daily_message(assessment, events)
        print(f"  → Sending to #{channel}")

    ok = send_slack(channel, blocks)
    print(f"  Slack: {'✓ sent' if ok else '✗ failed'}")


# Добавь в конец файла перед if __name__ == "__main__":
def run_test():
    """Test with fake anomaly events."""
    test_events = [
        {
            "template": "<HOST> hostd[<NUM>]: failed to connect to vpxa: connection refused",
            "predicted_label": "error",
            "risk_score": 10,
            "source": "cccluva-11-2.infra.local",
        },
        {
            "template": "<HOST> vmkernel: cpu<NUM> nmp error h:<HEX>",
            "predicted_label": "anomaly",
            "risk_score": 15,
            "source": "vmcluva-15f-7.infra.local",
        },
    ]
    print("Testing with fake events...")
    assessment = assess_severity(test_events)
    print(f"Severity: {assessment['severity']} | Urgent: {assessment['urgent']}")

    if assessment["urgent"]:
        channel = CHANNEL_URGENT
        blocks = build_urgent_message(assessment, test_events)
    else:
        channel = CHANNEL_DAILY
        blocks = build_daily_message(assessment, test_events)

    ok = send_slack(channel, blocks)
    print(f"Slack → #{channel}: {'✓ sent' if ok else '✗ failed'}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        run_test()
    else:
        run()
