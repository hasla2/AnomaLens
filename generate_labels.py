"""
generate_labels.py — auto-label log templates from collected events.

Reads all normalized event files from data/processed/events_*.jsonl,
deduplicates by template, assigns a label using rule-based heuristics,
and writes the labeled dataset to data/labels/labels.csv.

Label taxonomy (4 classes):
    noise          — routine heartbeats, API polling, known-good service events
    important      — VM/host lifecycle, config changes, migrations, snapshots
    error          — failures, exceptions, auth errors, timeouts, kernel warnings
    security_noise — login/logout, session tracking, certificate checks, scanners

Note: The "warning" class was removed due to insufficient labeled examples
(n=5 in production data). Templates previously labeled "warning" are
redistributed:
    - vmkwarning (kernel device warnings) → error
    - envoy config warnings (reuse_port)  → noise
    - envoy connection closing            → noise
    - vsansystem pywarning setup          → noise

A candidates file is produced for manual review of ambiguous
security_noise templates.
"""

import csv
import glob
import json
import os
import re
from collections import Counter

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_GLOB = "data/processed/events_*.jsonl"
OUTPUT_FILE = "data/labels/labels.csv"
CANDIDATES_FILE = "data/labels/label_candidates_security.csv"

# Supported labels — warning intentionally excluded (see module docstring)
LABELS = {"noise", "important", "error", "security_noise", "unknown"}

# ─── Template normalization ───────────────────────────────────────────────────


def normalize_template(template: str) -> str | None:
    """
    Apply secondary normalization to templates coming from the event files.
    Collapses API method names, UUIDs, long hex IDs, and numbers into
    typed placeholders to improve deduplication.
    """
    if not template:
        return None

    t = template.strip()

    # Collapse VMware/vSphere API method calls:
    # vim.RetrieveProperties → vim.<API_METHOD>
    t = re.sub(
        r"\b(vim|vpxapi|soap|csi|api|sdk|auth)\.[a-zA-Z0-9_.:-]+\b",
        r"\1.<API_METHOD>",
        t,
        flags=re.IGNORECASE,
    )

    # Quoted API method names
    t = re.sub(
        r'"(?:retrieveproperties|currenttime|waitforupdatesex|'
        r"acquirelocalticket|getview|query|update|create|delete|"
        r'destroy|reconfigure|poweron|poweroff|login|logout)"',
        '"<API_METHOD>"',
        t,
        flags=re.IGNORECASE,
    )

    # Standalone API method names
    t = re.sub(
        r"\b(?:retrieveproperties|currenttime|waitforupdatesex|"
        r"acquirelocalticket|getview|query|update|create|delete|"
        r"destroy|reconfigure|poweron|poweroff|login|logout)\b",
        "<API_METHOD>",
        t,
        flags=re.IGNORECASE,
    )

    # UUIDs
    t = re.sub(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
        r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "<UUID>",
        t,
    )

    # Long hex strings and IDs (≥12 hex chars)
    t = re.sub(r"\b[0-9a-fA-F]{12,}\b", "<ID>", t)

    # Integers
    t = re.sub(r"\b\d+\b", "<NUM>", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t


# ─── Rule-based labeling ──────────────────────────────────────────────────────

# Error indicators — patterns that reliably signal failures
_ERROR_KEYWORDS = [
    "error",
    "failed",
    "failure",
    "exception",
    "denied",
    "unauthorized",
    "timeout",
    "timed out",
    "cannot",
    "unable",
    "refused",
    "fatal",
    "critical",
    "invalid credentials",
    "authentication failed",
    "failed login",
    "unauthorized access",
    "vmkwarning",  # VMkernel-level warnings are operational errors
]

# Security-related patterns → security_noise
_SECURITY_KEYWORDS = [
    "login",
    "logout",
    "authentication",
    "authenticated",
    "session",
    "token",
]

# Operational change events → important
_IMPORTANT_EVENT_TYPES = {
    "operation_start",
    "migration_start",
    "config_change",
    "permission_change",
    "security_event",
    "task_start",
    "task_finish",
    "job_start",
    "job_finish",
}

_IMPORTANT_KEYWORDS = [
    "created",
    "deleted",
    "removed",
    "destroyed",
    "reconfigured",
    "changed",
    "modified",
    "updated",
    "assigned",
    "unassigned",
    "attached",
    "detached",
    "migrated",
    "cloned",
    "deployed",
    "powered on",
    "powered off",
    "reset",
    "snapshot",
    "permission",
    "role",
    "privilege",
    "alarm",
    "alert",
    "task",
    "job",
    "completed",
    "started",
    "finished",
]

# Known Windows infrastructure noise patterns
_INFRA_NOISE_KEYWORDS = [
    "wmi performance adapter",
    "background intelligent transfer service",
    "bits",
    "windows modules installer",
    "trustedinstaller",
    "volume shadow copy",
    "vss service",
    "microsoft storage spaces smp",
    "software protection",
    "sppsvc",
    "windows update medic service",
    "wuauserv",
    "windows insider service",
    "network setup service",
    "service control manager",
    "user profile service",
    "group policy",
    "dns client events",
    "time service",
    "dhcp-client",
]

# Routine API/service event types → noise
_NOISE_EVENT_TYPES = {
    "api_request",
    "task_success",
    "migration_success",
    "operation_finish",
    "service_start",
    "service_stop",
    "heartbeat",
    "status_check",
    "session_event",
}

# Patterns that contain "warn"/"warning" but are NOT operational warnings:
# they are either noise or errors handled by more specific rules above.
_WARN_NOISE_PATTERNS = [
    "pywarning",  # Python warnings module setup (vsansystem)
    "reuse_port",  # Envoy config advisory, not an operational issue
    "ignoreresourcewarning",  # Python warning filter config
]


def guess_label(template: str, event_type: str, service: str) -> str:
    """
    Assign a label to a log template using rule-based heuristics.

    Rule priority (highest to lowest):
      1. Errors / kernel warnings
      2. Security noise (Qualys scanner, login/logout)
      3. Important operational events
      4. Known infrastructure noise
      5. Routine API noise
      6. Default → noise
    """
    t = (template or "").lower()
    e = (event_type or "").lower()
    s = (service or "").lower()

    # ── 1. Errors ─────────────────────────────────────────────────────────────
    if e in ("error", "permission_error", "timeout", "failure"):
        return "error"

    if any(kw in t for kw in _ERROR_KEYWORDS):
        return "error"

    # ── 2. Security noise ─────────────────────────────────────────────────────
    if "qualys" in t or "qualys" in s:
        return "security_noise"

    if any(kw in t for kw in _SECURITY_KEYWORDS):
        return "security_noise"

    # ── 3. Noise patterns that contain "warn" but are not operational warnings ─
    # Must be checked before the important block to avoid false positives.
    if any(p in t for p in _WARN_NOISE_PATTERNS):
        return "noise"

    # ── 4. Known infrastructure noise ─────────────────────────────────────────
    if "sdrsinjector" in t or "sdrsinjector" in s:
        return "noise"

    if "opening slot count file" in t:
        return "noise"

    if "checking liveness" in t:
        return "noise"

    if e == "other" and "<datastore>" in t:
        return "noise"

    if any(kw in t for kw in _INFRA_NOISE_KEYWORDS):
        return "noise"

    if "<api_method>" in t:
        return "noise"

    # ── 5. Important operational events ───────────────────────────────────────
    if e in _IMPORTANT_EVENT_TYPES:
        return "important"

    if any(kw in t for kw in _IMPORTANT_KEYWORDS):
        return "important"

    # ── 6. Routine API/service events → noise ─────────────────────────────────
    if e in _NOISE_EVENT_TYPES:
        return "noise"

    # ── Default ───────────────────────────────────────────────────────────────
    return "noise"


def merge_labels(labels: list, min_confidence: float = 0.8) -> str:
    """
    Resolve conflicting labels for the same template across multiple events.
    Returns the dominant label if confidence ≥ min_confidence, else "unknown".
    """
    counter = Counter(labels)
    label, cnt = counter.most_common(1)[0]
    confidence = cnt / sum(counter.values())

    if len(counter) > 1 and confidence < min_confidence:
        return "unknown"

    return label


# ─── Helpers ──────────────────────────────────────────────────────────────────


def most_common_value(counter: Counter) -> str:
    return counter.most_common(1)[0][0] if counter else ""


def candidate_reason(template: str, event_type: str, service: str) -> str:
    """
    Return reasons why a template is a candidate for manual security_noise review.
    Used to populate the candidates CSV for human verification.
    """
    t = (template or "").lower()
    e = (event_type or "").lower()
    s = (service or "").lower()

    _SECURITY_CANDIDATE_KEYWORDS = [
        "login",
        "logout",
        "auth",
        "authentication",
        "token",
        "session",
        "certificate",
        "permission",
        "privilege",
        "denied",
        "unauthorized",
        "403",
        "notfound",
        "forbidden",
        "qualys",
        "vpxuser",
        "root",
        "administrator",
    ]

    reasons = []

    if e == "permission_error":
        reasons.append("event_type_permission_error")

    if any(kw in t for kw in _SECURITY_CANDIDATE_KEYWORDS) or any(
        kw in s for kw in _SECURITY_CANDIDATE_KEYWORDS
    ):
        reasons.append("security_keyword")

    return ",".join(reasons)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    files = glob.glob(INPUT_GLOB)
    if not files:
        print(f"No files found matching: {INPUT_GLOB}")
        return

    print(f"Processing {len(files)} event file(s)...")
    rows: dict[str, dict] = {}

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                event = json.loads(line)
                template = normalize_template(event.get("template"))
                if not template:
                    continue

                label = guess_label(
                    template,
                    event.get("event_type"),
                    event.get("service"),
                )
                if label not in LABELS:
                    label = "unknown"

                if template not in rows:
                    rows[template] = {
                        "template": template,
                        "labels_seen": [],
                        "event_types": Counter(),
                        "services": Counter(),
                        "platforms": Counter(),
                        "count": 0,
                    }

                rows[template]["labels_seen"].append(label)
                rows[template]["event_types"][event.get("event_type") or ""] += 1
                rows[template]["services"][event.get("service") or ""] += 1
                rows[template]["platforms"][event.get("platform") or ""] += 1
                rows[template]["count"] += 1

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(CANDIDATES_FILE), exist_ok=True)

    output_rows = []
    candidate_rows = []

    for row in rows.values():
        labels_counter = Counter(row["labels_seen"])
        final_label = merge_labels(row["labels_seen"])

        output_rows.append(
            {
                "template": row["template"],
                "label": final_label,
                "event_type": most_common_value(row["event_types"]),
                "service": most_common_value(row["services"]),
                "platform": most_common_value(row["platforms"]),
                "count": row["count"],
                "label_confidence": round(
                    labels_counter.most_common(1)[0][1] / sum(labels_counter.values()),
                    3,
                ),
                "label_candidates": json.dumps(
                    dict(labels_counter), ensure_ascii=False
                ),
            }
        )

        reason = candidate_reason(
            row["template"],
            most_common_value(row["event_types"]),
            most_common_value(row["services"]),
        )
        if reason:
            candidate_rows.append(
                {
                    "template": row["template"],
                    "suggested_label": final_label,
                    "reason": reason,
                    "event_type": most_common_value(row["event_types"]),
                    "service": most_common_value(row["services"]),
                    "platform": most_common_value(row["platforms"]),
                    "count": row["count"],
                    "label_confidence": round(
                        labels_counter.most_common(1)[0][1]
                        / sum(labels_counter.values()),
                        3,
                    ),
                    "label_candidates": json.dumps(
                        dict(labels_counter), ensure_ascii=False
                    ),
                }
            )

    # Write labeled dataset (sorted by frequency, most common first)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "template",
                "label",
                "event_type",
                "service",
                "platform",
                "count",
                "label_confidence",
                "label_candidates",
            ],
        )
        writer.writeheader()
        for row in sorted(output_rows, key=lambda x: x["count"], reverse=True):
            writer.writerow(row)

    # Write security candidates for manual review
    with open(CANDIDATES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "template",
                "suggested_label",
                "reason",
                "event_type",
                "service",
                "platform",
                "count",
                "label_confidence",
                "label_candidates",
            ],
        )
        writer.writeheader()
        for row in sorted(candidate_rows, key=lambda x: x["count"], reverse=True):
            writer.writerow(row)

    print(f"Created {OUTPUT_FILE} with {len(output_rows)} templates")
    print(f"Created {CANDIDATES_FILE} with {len(candidate_rows)} security candidates")

    # Summary statistics
    label_counts = Counter(r["label"] for r in output_rows)
    print("\nLabel distribution:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / len(output_rows)
        print(f"  {lbl:<16} {cnt:>6}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
