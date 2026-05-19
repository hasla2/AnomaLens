"""
scheduler.py — AnomaLens scheduler.

Runs the full pipeline on a schedule:
    Every 15 minutes  : collect logs → classify → RAG for anomalies
    Every 12 hours    : generate infrastructure summary report

Usage:
    # Start scheduler (runs until Ctrl+C):
    python scheduler.py

    # Run collection once immediately (for testing):
    python scheduler.py --once

Requirements:
    pip install apscheduler
    Ollama must be running for RAG and summary:
        ollama serve
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.blocking import BlockingScheduler

# ─── Configuration ────────────────────────────────────────────────────────────

PYTHON = sys.executable
COLLECT_SCRIPT = "src/collector/graylog_collect.py"
PREDICT_SCRIPT = "predict.py"
SUMMARY_SCRIPT = "src/summary/daily_report.py"
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("data/reports")
LOGS_DIR = Path("logs")

# Number of top anomalies to explain via RAG per collection cycle
MAX_RAG_ANOMALIES = 3

# ─── Logging ──────────────────────────────────────────────────────────────────

LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "scheduler.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("anomalens")

# ─── Helpers ──────────────────────────────────────────────────────────────────


def predictions_files_last_12h() -> list[Path]:
    """
    Return all predictions_<ts>.jsonl files whose timestamp is within
    the last 12 hours, sorted oldest → newest.

    Filename format: predictions_20260514T063729Z.jsonl
    """
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
    result = []
    for p in sorted(PROCESSED_DIR.glob("predictions_*.jsonl")):
        ts_str = p.stem.replace("predictions_", "")  # 20260514T063729Z
        try:
            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
            if ts >= cutoff:
                result.append(p)
        except ValueError:
            continue
    return result


# ─── Pipeline steps ───────────────────────────────────────────────────────────


def run_script(script: str, label: str) -> bool:
    """Run a Python script as subprocess. Returns True on success."""
    log.info(f"[{label}] Starting: {script}")
    try:
        result = subprocess.run(
            [PYTHON, script],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per step
        )
        if result.returncode == 0:
            log.info(f"[{label}] Done")
            return True
        else:
            log.error(f"[{label}] Failed (exit {result.returncode})")
            if result.stderr:
                log.error(f"[{label}] stderr: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"[{label}] Timeout after 10 minutes")
        return False
    except Exception as e:
        log.error(f"[{label}] Exception: {e}")
        return False


def run_rag_for_anomalies():
    """
    Called as part of the 12-hour summary job.
    Reads ALL predictions_<ts>.jsonl files from the last 12 hours,
    deduplicates anomalies/errors by template, and explains the top N via RAG.
    """
    files = predictions_files_last_12h()
    if not files:
        log.warning("[RAG] No predictions files found for the last 12 hours — skipping")
        return

    log.info(f"[RAG] Reading {len(files)} file(s) from the last 12 hours")

    try:
        events = []
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    if e.get("predicted_label") in ("error", "anomaly"):
                        events.append(e)

        if not events:
            log.info("[RAG] No anomalies detected in the last 12 hours")
            return

        log.info(f"[RAG] {len(events)} anomaly event(s) found across all files")

        # Deduplicate by template, take top N by risk score
        unique = {e["template"]: e for e in events}
        top = sorted(
            unique.values(), key=lambda x: x.get("risk_score", 0), reverse=True
        )[:MAX_RAG_ANOMALIES]

        log.info(f"[RAG] Explaining {len(top)} anomaly(ies)...")

        from src.rag.retriever import RAGRetriever

        rag = RAGRetriever()

        for event in top:
            result = rag.explain(event["template"])
            host = event.get("source", event.get("host", "unknown"))
            log.info(
                f"[RAG] {host} | {event['template'][:60]}...\n"
                f"      → {result['explanation'][:150]}"
            )

    except Exception as e:
        log.warning(f"[RAG] Unavailable: {e}")


# ─── Scheduled jobs ───────────────────────────────────────────────────────────


def job_collect_and_predict():
    """
    Runs every 15 minutes:
      1. Collect new logs from Graylog
      2. Classify all events with LinearSVC
      3. Run RAG explanation for detected anomalies
    """
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    log.info(f"{'─'*50}")
    log.info(f"Collection cycle started at {ts} UTC")

    # Step 1: Collect
    ok = run_script(COLLECT_SCRIPT, "COLLECT")
    if not ok:
        log.error("Collection failed — skipping classify and RAG")
        return

    # Step 2: Classify
    ok = run_script(PREDICT_SCRIPT, "CLASSIFY")
    if not ok:
        log.error("Classification failed — skipping RAG")
        return

    # Step 3: RAG для свежих аномалий (только из последнего файла)
    run_rag_for_latest()

    log.info("Collection cycle complete")

    # Step 4: LLM decides: urgent → #alerts-critical, else → #alerts-daily
    run_script("notify.py", "NOTIFY")


def run_rag_for_latest():
    """RAG for anomalies from the most recent predictions file only."""
    files = sorted(PROCESSED_DIR.glob("predictions_*.jsonl"))
    if not files:
        return

    latest = files[-1]
    events = []
    with open(latest, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("predicted_label") in ("error", "anomaly"):
                events.append(e)

    if not events:
        log.info("[RAG] No anomalies in latest batch")
        return

    # Only top-1 to keep within 15-min window
    top = sorted(events, key=lambda x: x.get("risk_score", 0), reverse=True)[:1]

    try:
        from src.rag.retriever import RAGRetriever

        rag = RAGRetriever()
        for event in top:
            result = rag.explain(event["template"])
            log.info(f"[RAG] {event.get('source','?')} | {event['template'][:60]}")
            log.info(f"      → {result['explanation'][:200]}")
    except Exception as e:
        log.warning(f"[RAG] Unavailable: {e}")


def job_summary():
    """
    Runs every 12 hours:
      Aggregate all events from the past 12 hours and generate
      a Markdown report via Mistral-7B.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info(f"{'─'*50}")
    log.info(f"Summary started at {ts}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # RAG explanation for anomalies accumulated over the last 12 hours
    run_rag_for_anomalies()

    # Generate Markdown summary report
    run_script(SUMMARY_SCRIPT, "SUMMARY")
    log.info("Summary complete")


# ─── Scheduler event listener ─────────────────────────────────────────────────


def on_job_event(event):
    if event.exception:
        log.error(f"Job {event.job_id} raised an exception: {event.exception}")


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="AnomaLens scheduler")
    p.add_argument(
        "--once",
        action="store_true",
        help="Run one collection cycle immediately and exit",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Run one summary cycle immediately and exit",
    )
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 50)
    log.info("AnomaLens Scheduler starting")
    log.info(f"  Collect script : {COLLECT_SCRIPT}")
    log.info(f"  Predict script : {PREDICT_SCRIPT}")
    log.info(f"  Summary script : {SUMMARY_SCRIPT}")
    log.info("=" * 50)

    # One-shot modes for testing
    if args.once:
        log.info("Running one collection cycle (--once mode)")
        job_collect_and_predict()
        log.info("Done.")
        return

    if args.summary:
        log.info("Running one summary cycle (--summary mode)")
        job_summary()
        log.info("Done.")
        return

    # Scheduler
    scheduler = BlockingScheduler(timezone="UTC")

    # Run collection immediately on startup, then every 15 minutes
    scheduler.add_job(
        job_collect_and_predict,
        trigger="interval",
        minutes=15,
        id="collect",
        name="Collect + Classify + RAG",
        next_run_time=datetime.now(timezone.utc),  # run immediately at start
    )

    # Summary every 12 hours
    scheduler.add_job(
        job_summary,
        trigger="interval",
        hours=12,
        id="summary",
        name="12h Summary",
    )

    scheduler.add_listener(on_job_event, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    log.info("Scheduler running. Press Ctrl+C to stop.")
    log.info("  • Collect + Classify + RAG : every 15 minutes")
    log.info("  • Summary                  : every 12 hours")
    log.info("─" * 50)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user.")


if __name__ == "__main__":
    main()
