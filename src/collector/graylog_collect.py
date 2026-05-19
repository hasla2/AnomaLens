"""
graylog_collect.py — collect and normalize log events from Graylog.

Queries the Graylog REST API every 15 minutes (via scheduler.py),
deduplicates events by message ID, normalizes each message into a
structured template, and writes results to JSONL files.

Output:
    data/raw/<platform>_<timestamp>.jsonl       raw Graylog messages
    data/processed/events_<timestamp>.jsonl     normalized events
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import urllib3
import yaml
from normalizer import (detect_event_type, detect_service, normalize_hosts,
                        normalize_template)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def graylog_search(
    base_url: str,
    token: str,
    query: str,
    range_seconds: int = 900,
    limit: int = 10000,
    verify_tls: bool = True,
) -> dict:
    """
    Execute a relative-time search against the Graylog REST API.

    Args:
        base_url:       Graylog base URL (e.g. https://graylog.example.com/api)
        token:          Graylog API token
        query:          Graylog search query string
        range_seconds:  Time window in seconds (default: 900 = 15 min)
        limit:          Maximum number of messages to retrieve
        verify_tls:     Whether to verify TLS certificates

    Returns:
        Parsed JSON response from Graylog.
    """
    url = f"{base_url.rstrip('/')}/search/universal/relative"
    params = {
        "query": query,
        "range": range_seconds,
        "limit": limit,
        "sort": "timestamp:desc",
        "fields": (
            "timestamp,source,message,level,"
            "facility,facility_num,gl2_remote_ip,streams"
        ),
    }
    response = requests.get(
        url,
        params=params,
        auth=(token, "token"),
        headers={"Accept": "application/json"},
        verify=verify_tls,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def normalize_event(raw_message: dict, platform: str) -> dict:
    """
    Convert a raw Graylog message into a normalized event record.

    Normalization steps (delegated to normalizer.py):
      - Replace hostnames with <HOST>
      - Replace variable fields (IPs, numbers, UUIDs, hex) with typed placeholders
      - Detect the originating service and event type

    Args:
        raw_message: A single message object from the Graylog API response.
        platform:    Source platform identifier ("esxi" or "hyperv").

    Returns:
        Structured event dict ready for classification and storage.
    """
    m = raw_message.get("message", {})
    message = m.get("message", "")

    # Normalize hostnames before further processing
    normalized_message = normalize_hosts(message)

    return {
        "id": m.get("gl2_message_id") or m.get("_id") or raw_message.get("_id"),
        "timestamp": m.get("timestamp"),
        "platform": platform,
        "host": normalize_hosts(m.get("source", "")),
        "cluster": None,
        "source": m.get("source"),  # original (non-anonymized) source
        "service": detect_service(message),
        "severity": m.get("level"),
        "facility": m.get("facility"),
        "message": message,  # raw message preserved for debugging
        "template": normalize_template(normalized_message),
        "event_type": detect_event_type(message),
        "label": None,  # filled later by predict.py
    }


def write_jsonl(path: str, rows: list) -> None:
    """Append a list of dicts to a JSONL file, creating parent dirs if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = load_config()
    base_url = cfg["graylog"]["url"]
    token = cfg["graylog"]["token"]
    verify_tls = cfg["graylog"].get("verify_tls", True)
    range_seconds = cfg["collector"].get("range_seconds", 900)
    limit = cfg["collector"].get("limit", 10000)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seen_ids = set()  # deduplicate across platforms within one run

    for platform, query in cfg["queries"].items():
        data = graylog_search(base_url, token, query, range_seconds, limit, verify_tls)
        raw_messages = data.get("messages", [])
        normalized = []

        for x in raw_messages:
            event = normalize_event(x, platform)
            if event["id"] in seen_ids:
                continue
            seen_ids.add(event["id"])
            normalized.append(event)

        # Write raw messages for audit trail
        write_jsonl(f"data/raw/{platform}_{run_ts}.jsonl", raw_messages)

        # Write normalized events for classification
        write_jsonl(f"data/processed/events_{run_ts}.jsonl", normalized)

        print(f"{platform}: collected={len(raw_messages)}, unique={len(normalized)}")


if __name__ == "__main__":
    main()
