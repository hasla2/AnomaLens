import re

# ── Host normalization ─────────────────────────────────────────────────────────

HOST_RE = re.compile(
    r"\b(?:"
    r"HOST_NAME_PREFIX[a-z0-9]*-\d+[a-z0-9]*-\d+|"
    r")"
    r"(?:\.(?:DOMAIN1\.COM|DOMAIN2\.NET))?"
    r"\b",
    re.IGNORECASE,
)


def normalize_hosts(text):
    if not text:
        return text
    return HOST_RE.sub("<HOST>", text.lower())


# ── Datastore normalization ────────────────────────────────────────────────────

DATASTORE_RE = re.compile(r"(/vmfs/volumes//)[^/\s]+", re.IGNORECASE)
DATASTORE_VOL_RE = re.compile(r"\bvol\s+'[^']+'", re.IGNORECASE)


def normalize_datastores(text):
    if not text:
        return text
    text = DATASTORE_RE.sub(r"\1<DATASTORE>", text)
    text = re.sub(r"\.naa\.[^/\s]+", ".naa.<ID>", text, flags=re.IGNORECASE)
    text = DATASTORE_VOL_RE.sub("vol '<DATASTORE>'", text)
    return text


# ── Placeholder normalization ──────────────────────────────────────────────────


def normalize_placeholders(text):
    if not text:
        return text
    return (
        text.replace("<host>", "<HOST>")
        .replace("<num>", "<NUM>")
        .replace("<id>", "<ID>")
        .replace("<uuid>", "<UUID>")
        .replace("<ip>", "<IP>")
        .replace("<hex>", "<HEX>")
        .replace("<timestamp>", "<TIMESTAMP>")
        .replace("<datastore>", "<DATASTORE>")
    )


# ── Template normalization ─────────────────────────────────────────────────────


def normalize_template(message):
    msg = normalize_hosts(message)
    msg = normalize_datastores(msg)
    msg = msg.lower()

    msg = re.sub(r"\b\d{4}-\d{2}-\d{2}t[\d:.]+z\b", "<timestamp>", msg)
    msg = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<uuid>", msg)
    msg = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", msg)
    msg = re.sub(r"0x[0-9a-f]+", "<hex>", msg)
    msg = re.sub(r"\bcpu\d+\b", "cpu<NUM>", msg)
    msg = re.sub(r"\bdlx:\s*\d+", "dlx:<NUM>", msg)
    msg = re.sub(r"\bopid=[a-z0-9-]+\b", "opid=<id>", msg)
    msg = re.sub(r"\bsid=[a-z0-9-]+\b", "sid=<id>", msg)
    msg = re.sub(r"\blro-\d+\b", "lro-<num>", msg)
    msg = re.sub(r"\btask-\d+\b", "task-<num>", msg)
    msg = re.sub(r"\b\d+\b", "<num>", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    msg = normalize_placeholders(msg)
    return msg


# ── Service detection ──────────────────────────────────────────────────────────

_SERVICE_PATTERNS = [
    "Hostd",
    "Vpxa",
    "Fdm",
    "healthd",
    "healthdPlugins",
    "sdrsInjector",
    "sandboxd",
    "esxtokend",
    "kmxa",
    "envoy-access",
    "VMMS",
    "VMSP",
    "Hyper-V",
    "FailoverClustering",
    "WMI Performance Adapter",
    "Background Intelligent Transfer Service",
    "VMM",
    "WinRM",
    "Windows Error Reporting",
    "Service Control Manager",
    "Microsoft-Windows-Hyper-V-VMMS",
    "Microsoft-Windows-Hyper-V-Worker",
    "Microsoft-Windows-FailoverClustering",
    "vsansystem",
    "vSAN",
]

_WIN_SERVICE_RE = [
    re.compile(r"The (.+?) service entered the"),
    re.compile(r"Successfully scheduled (.+?) service"),
    re.compile(r"The start type of the (.+?) se"),
]


def detect_service(message):
    for pattern in _WIN_SERVICE_RE:
        m = pattern.search(message)
        if m:
            return m.group(1).strip()

    msg = message.lower()
    for p in _SERVICE_PATTERNS:
        if p.lower() in msg:
            return p

    if "hyper-v" in msg:
        return "Hyper-V"
    if "failoverclustering" in msg or "failover clustering" in msg:
        return "FailoverClustering"

    return "unknown"


# ── Event type detection ───────────────────────────────────────────────────────


def detect_event_type(message):
    msg = message.lower()

    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "notfound" in msg or "not found" in msg or "403" in msg or "denied" in msg:
        return "permission_error"
    if "failed" in msg or "failure" in msg or "error" in msg or "exception" in msg:
        return "error"
    if "heartbeat" in msg or "heart beat" in msg:
        return "heartbeat"
    if "entered the running state" in msg or "started" in msg:
        return "service_start"
    if "entered the stopped state" in msg or "stopped" in msg:
        return "service_stop"
    if "created" in msg or "task created" in msg:
        return "task_created"
    if "completed" in msg or "success" in msg or "status success" in msg:
        return "task_success"
    if "post /hgw/" in msg or "post /sdk" in msg or "get /hgw/" in msg:
        return "api_request"
    if "access history in hive" in msg:
        return "registry_activity"
    if "service entered the running" in msg:
        return "service_start"
    if "service entered the stopped" in msg:
        return "service_stop"
    if "start type of" in msg:
        return "service_config_change"
    if "offline downlevel migration succeeded" in msg:
        return "migration_success"

    return "other"
