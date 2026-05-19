"""
evaluate.py — unified evaluation script for AnomaLens.

Loads pre-trained classical classifiers from models/ and compares them
against optional LLM-based models. All results are saved to data/results/.

Prerequisites:
    Run train_model.py first to train and save classical classifiers

    For --llm and --mistral: Ollama must be running:
        ollama serve

    For --logbert: fine-tuned BERT model must exist at:
        models/logbert_anomalens_final/   (fine-tune on Google Colab first, use LogBERT_fine_tuning.ipynb)

Classical classifiers loaded from models/:
    Naive Bayes (CNB)   — models/naive_bayes.joblib
    Random Forest       — models/random_forest.joblib
    LogReg (baseline)   — models/logreg.joblib
    LinearSVC           — models/linearsvc.joblib   ← was chosed as default

Optional LLM models (require additional setup):
    Phi-3.5-Mini zero-shot   — via Ollama, GGUF (--llm)
    Mistral7B                — via Ollama, GGUF (--mistral)
    LogBERT fine-tuned       — via HuggingFace transformers (--logbert)

Usage:
    # Load and compare all classical classifiers (~5 sec):
    python evaluate.py

    # Classical classifiers + 5-fold cross-validation:
    python evaluate.py --crossval

    # Classical classifiers + Phi-3.5-Mini zero-shot (requires Ollama):
    python evaluate.py --llm --llm-limit 30

    # Classical classifiers + Mistral-7B zero-shot (requires Ollama):
    python evaluate.py --mistral --mistral-limit 30

    # Classical classifiers + fine-tuned LogBERT:
    python evaluate.py --logbert

    # Full evaluation — all models + cross-validation + save predictions:
    python evaluate.py --crossval --mistral --logbert --save-predictions  // --llm this excluded as Phi fails at zero-shot classification

Arguments:
    --crossval              Run 5-fold stratified cross-validation for all classical classifiers
    --llm                   Evaluate Phi-3.5-Mini zero-shot via Ollama
    --mistral               Evaluate Mistral7b via Ollama
    --logbert               Evaluate fine-tuned LogBERT (requires models/logbert_anomalens_final/)
    --llm-limit N           Max examples for LLM (Phi) evaaluation (default: 30)
    --mistral-limit N       Max examples for Mistral evaaluation (default: 30)
    --save-predictions      Save per-example predictions to CSV

Output files saved to data/results/:
    summary_<id>.csv        comparison table: all models × metrics
    crossval_<id>.json      5-fold CV results with 95% confidence intervals
    llm_details_<id>.jsonl  per-example LLM predictions and latency
    run_meta_<id>.json      run metadata for reproducibility
    predictions_<id>.csv    per-example predictions (only with --save-predictions)
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

LABELS_FILE = "data/labels/labels.csv"
RESULTS_DIR = Path("data/results")
MODELS_DIR = Path("models")
ALL_LABELS = ["noise", "important", "error", "security_noise"]
SEARCH_ORDER = ["security_noise", "error", "important", "noise"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
OOD_THRESHOLD = 0.45  # predictions below this confidence are flagged as anomaly

# Terminal color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ─── LLM system prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Classify the log template. Reply with ONLY this JSON, nothing else:
{"label": "<label>", "confidence": 0.9, "explanation": "<reason>", "severity": 5}

Labels: noise, important, error, security_noise"""

# ─── Mistral LLM system prompt ────────────────────────────────────────────────────────

MISTRAL_PROMPT = """\
You are a VMware/Hyper-V log classifier. Reply with ONE WORD only.

Labels meaning:
- noise: routine heartbeats, API polling, status checks, background services
- important: VM lifecycle events (power on/off, migration, snapshot, config change)
- error: failures, timeouts, connection refused, exceptions, fatal errors
- security_noise: login, logout, authentication, session, token events

Examples:
Template: <host> sdrsinjector[<num>]: opening slot count file
Label: noise

Template: <host> hostd[<num>]: checking liveness of
Label: noise

Template: <host> hostd[<num>]: failed to connect to vpxa: connection refused
Label: error

Template: <host> envoy[<num>]: upstream request timeout
Label: error

Template: <host> vmms[<num>]: virtual machine powered on
Label: important

Template: <host> vpxd[<num>]: vm migrated successfully
Label: important

Template: vpxuser login from <ip>
Label: security_noise

Template: <host> envoy[<num>]: successful authentication session
Label: security_noise"""

# ─── Data loading ─────────────────────────────────────────────────────────────


def load_dataset(path: str) -> pd.DataFrame:
    """Load and filter the labeled template dataset."""
    df = pd.read_csv(path)
    df["template"] = df["template"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df = df[df["template"] != ""]
    df = df[df["label"].isin(ALL_LABELS)]
    return df


def split_dataset(df: pd.DataFrame):
    """Stratified 80/20 train/test split with fixed random seed."""
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )


# ─── Classifier pipelines ─────────────────────────────────────────────────────


def build_pipeline(clf) -> Pipeline:
    """Wrap a classifier in a TF-IDF (1,2)-gram pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", clf),
        ]
    )


# All classifiers share the same TF-IDF feature extraction.
# class_weight="balanced" compensates for the severe label imbalance
# (noise accounts for ~78.8% of all templates).
CLASSIFIERS = {
    "Naive Bayes (CNB)": ComplementNB(),
    "Random Forest": RandomForestClassifier(
        100, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    ),
    "LogReg (baseline)": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "LinearSVC": LinearSVC(max_iter=2000, class_weight="balanced"),
}

# ─── LLM evaluator (Phi-3.5-Mini via Ollama) ──────────────────────────────────


class LLMEvaluator:
    """
    Evaluates Phi-3.5-Mini on log template classification via Ollama.

    Design note: Mistral-7B and NeMo 12B were initially tested for LLM-based
    classification but excluded due to prohibitive CPU inference latency
    (>180s and >1000s per example respectively). Phi-3.5-Mini (2.4 GB,
    ~120s/item on CPU) was retained for RAG explanations where latency is
    less critical.
    """

    def __init__(self):
        # Bypass corporate proxy for localhost Ollama connections
        import ollama

        self._ollama = ollama
        self.model_name = "phi35"
        self.name = "Phi-3.5-Mini (zero-shot, Ollama)"

    def _call(self, template: str) -> dict:
        """Send one template to the LLM and parse the JSON response."""
        resp = self._ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify:\n{template[:300]}"},
            ],
            format="json",
            options={"temperature": 0.05, "num_predict": 150, "keep_alive": "10m"},
        )
        # Support both dict-style and Pydantic-style Ollama response objects
        try:
            raw = resp["message"]["content"].strip()
        except (TypeError, KeyError):
            raw = resp.message.content.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            result = json.loads(m.group()) if m else {}

        # Phi-3.5 sometimes returns {"noise": "..."} instead of {"label": "noise"}
        label = result.get("label", "unknown")
        if label not in ALL_LABELS + ["unknown"]:
            for key in result.keys():
                if key in ALL_LABELS:
                    label = key
                    break

        result["label"] = label
        return result

    def predict(self, templates: list, limit: int = 30) -> tuple:
        """
        Run inference on up to `limit` templates.
        Returns (labels, confidences, inference_times_per_item).
        """
        labels, confs, times = [], [], []
        total = min(len(templates), limit)

        for i, tpl in enumerate(templates[:total]):
            t0 = time.time()
            try:
                result = self._call(tpl)
            except Exception:
                result = {"label": "unknown", "confidence": 0.0}

            elapsed = time.time() - t0
            labels.append(result.get("label", "unknown"))
            # confs.append(float(result.get("confidence", 0.5)))
            confs.append(float(result.get("confidence") or 0.5))
            times.append(elapsed)

            # Inline progress bar
            pct = 100 * (i + 1) // total
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {i+1}/{total}  {elapsed:.0f}s/item", end="", flush=True)

        print()
        return labels, confs, times


# ─── Mistral-7B evaluator ─────────────────────────────────────────────────────


class MistralEvaluator:
    """
    Evaluates Mistral-7B-Instruct on log template classification via Ollama.
    Mistral follows instructions better than Phi-3.5-Mini for this task.
    """

    def __init__(self):
        import ollama

        self._ollama = ollama
        self.model_name = "mistral7b"
        self.name = "Mistral-7B (zero-shot, Ollama)"

    def _call(self, template: str) -> str:
        resp = self._ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": MISTRAL_PROMPT},
                {"role": "user", "content": f"Template: {template[:300]}\nLabel:"},
            ],
            options={"temperature": 0.0, "num_predict": 15, "keep_alive": "10m"},
        )
        try:
            raw = resp["message"]["content"].strip().lower()
        except (TypeError, KeyError):
            raw = resp.message.content.strip().lower()

        for lbl in SEARCH_ORDER:
            pattern = lbl.replace("_", r"[_\s]")
            if re.search(r"\b" + pattern + r"\b", raw):
                return lbl
        return "unknown"

    def predict(self, templates: list, limit: int = 30) -> tuple:
        """Run inference on up to `limit` templates."""
        labels, times = [], []
        total = min(len(templates), limit)

        for i, tpl in enumerate(templates[:total]):
            t0 = time.time()
            try:
                label = self._call(tpl)
            except Exception:
                label = "unknown"

            elapsed = time.time() - t0
            labels.append(label)
            times.append(elapsed)

            pct = 100 * (i + 1) // total
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {i+1}/{total}  {elapsed:.0f}s/item", end="", flush=True)

        print()
        confs = [0.5] * len(labels)
        return labels, confs, times


# ─── LogBERT evaluator (fine-tuned BERT) ──────────────────────────────────────


class LogBERTEvaluator:
    """
    Evaluates fine-tuned BERT (LogBERT-style) on log template classification.
    Model must be fine-tuned on Colab and saved to models/logbert_anomalens_final/.
    """

    def __init__(self, model_path: str = "models/logbert_anomalens_final"):
        from transformers import pipeline as hf_pipeline

        print(f"  Loading LogBERT from {model_path}...")
        self.classifier = hf_pipeline(
            "text-classification",
            model=model_path,
            device=-1,  # CPU
            truncation=True,
            max_length=128,
        )
        self.name = "LogBERT (fine-tuned)"

    def predict(self, templates: list, limit: int = None) -> tuple:
        """Run inference on templates. Returns (labels, confidences, times)."""
        labels, confs, times = [], [], []
        total = min(len(templates), limit) if limit else len(templates)

        for i, tpl in enumerate(templates[:total]):
            t0 = time.time()
            result = self.classifier(tpl[:512])[0]
            elapsed = time.time() - t0

            label = result["label"].lower()
            if label not in ALL_LABELS:
                label = "unknown"

            labels.append(label)
            confs.append(round(float(result["score"]), 3))
            times.append(elapsed)

            pct = 100 * (i + 1) // total
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {i+1}/{total}  {elapsed:.2f}s/item", end="", flush=True)

        print()
        return labels, confs, times


# ─── Metrics helpers ──────────────────────────────────────────────────────────


def metrics_row(name: str, y_true: list, y_pred: list) -> dict:
    """Compute a summary metrics row for the comparison table."""
    return {
        "Model": name,
        "Accuracy": round(float(np.mean(np.array(y_true) == np.array(y_pred))), 4),
        "Macro F1": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "W. F1": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "F1 error": round(
            f1_score(
                y_true, y_pred, labels=["error"], average="micro", zero_division=0
            ),
            4,
        ),
        "F1 sec_n": round(
            f1_score(
                y_true,
                y_pred,
                labels=["security_noise"],
                average="micro",
                zero_division=0,
            ),
            4,
        ),
    }


def run_crossval(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    5-fold stratified cross-validation for all classifiers.
    Returns macro F1 mean, std, and 95% CI per classifier.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X, y = df["template"].tolist(), df["label"].tolist()
    results = {}

    for clf_name, clf in CLASSIFIERS.items():
        pipe = build_pipeline(clf)
        f1s = []
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            pipe.fit([X[i] for i in tr], [y[i] for i in tr])
            y_pred = pipe.predict([X[i] for i in va])
            f1 = f1_score([y[i] for i in va], y_pred, average="macro", zero_division=0)
            f1s.append(f1)
            print(f"    {clf_name:<22}  fold {fold}/{n_splits}: F1={f1:.3f}")

        arr = np.array(f1s)
        results[clf_name] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "ci95": (
                round(float(arr.mean() - 1.96 * arr.std()), 4),
                round(float(arr.mean() + 1.96 * arr.std()), 4),
            ),
            "folds": [round(f, 4) for f in f1s],
        }
    return results


# ─── Console output ───────────────────────────────────────────────────────────


def print_header(text: str):
    print(f"\n{BOLD}{BLUE}{'─'*60}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'─'*60}{RESET}")


def print_comparison_table(rows: list):
    """Print the comparison table; best model is shown in bold."""
    cols = ["Model", "Accuracy", "Macro F1", "W. F1", "F1 error", "F1 sec_n"]
    col_w = max(len(r["Model"]) for r in rows) + 2
    # fmt   = "{:<{w}} {:>9} {:>9} {:>8} {:>9} {:>8} {:>8}"
    fmt = "{:<{w}} {:>9} {:>9} {:>8} {:>9} {:>8}"
    print(fmt.format(*cols, w=col_w))
    print("─" * (col_w + 62))
    for r in rows:
        is_best = r.pop("_best", False)
        line = fmt.format(
            r["Model"],
            r["Accuracy"],
            r["Macro F1"],
            r["W. F1"],
            r["F1 error"],
            r["F1 sec_n"],
            w=col_w,
        )
        print(f"{BOLD}{line}{RESET}" if is_best else line)


def print_detail_report(name: str, y_true: list, y_pred: list, labels: list):
    """Per-class precision / recall / F1 and confusion matrix."""
    print_header(f"Detailed report: {name}")
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    row_fmt = "{:<18} {:>10} {:>10} {:>10} {:>8}"
    print(row_fmt.format("Class", "Precision", "Recall", "F1", "Support"))
    print("─" * 62)
    for lbl in labels:
        if lbl not in rep:
            continue
        d = rep[lbl]
        f1 = d["f1-score"]
        color = GREEN if f1 >= 0.8 else (YELLOW if f1 >= 0.5 else RED)
        print(
            row_fmt.format(
                lbl,
                f"{d['precision']:.3f}",
                f"{d['recall']:.3f}",
                f"{color}{f1:.3f}{RESET}",
                int(d["support"]),
            )
        )

    print("\nConfusion matrix (rows = true label, cols = predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(" " * 16 + "  ".join(f"{l[:8]:>10}" for l in labels))
    for i, row in enumerate(cm):
        vals = "  ".join(
            f"{BOLD}{v:>10}{RESET}" if i == j else f"{v:>10}" for j, v in enumerate(row)
        )
        print(f"  {labels[i]:<14}  {vals}")


# ─── Save results ─────────────────────────────────────────────────────────────


def save_results(rows: list, cv_results: dict, llm_details: list, run_id: str, args):
    """Save summary CSV, cross-validation JSON, LLM details, and run metadata."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Main comparison table → used directly in the report
    summary_path = RESULTS_DIR / f"summary_{run_id}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\n{GREEN}Saved:{RESET} {summary_path}")

    # Cross-validation results
    if cv_results:
        cv_path = RESULTS_DIR / f"crossval_{run_id}.json"
        with open(cv_path, "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"{GREEN}Saved:{RESET} {cv_path}")

    # LLM per-example details for qualitative analysis in the report
    if llm_details:
        llm_path = RESULTS_DIR / f"llm_details_{run_id}.jsonl"
        with open(llm_path, "w", encoding="utf-8") as f:
            for d in llm_details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"{GREEN}Saved:{RESET} {llm_path}")

    # Run metadata for reproducibility
    meta_path = RESULTS_DIR / f"run_meta_{run_id}.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "args": vars(args),
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "ood_threshold": OOD_THRESHOLD,
            },
            f,
            indent=2,
        )
    print(f"{GREEN}Saved:{RESET} {meta_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="AnomaLens — unified classifier and LLM evaluation"
    )
    p.add_argument("--labels", default=LABELS_FILE, help="Path to labels.csv")
    p.add_argument(
        "--crossval",
        action="store_true",
        help="Run 5-fold cross-validation for all classifiers",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="Include Phi-3.5-Mini via Ollama in evaluation",
    )
    p.add_argument(
        "--logbert",
        action="store_true",
        help="Fine-tuned LogBERT (requires models/logbert_anomalens_final/)",
    )
    p.add_argument(
        "--mistral",
        action="store_true",
        help="Evaluate Mistral-7B zero-shot via Ollama",
    )
    p.add_argument(
        "--llm-limit",
        type=int,
        default=30,
        help="Max examples for LLM evaluation (default: 30)",
    )
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-example predictions to CSV",
    )
    p.add_argument(
        "--mistral-limit",
        type=int,
        default=30,
        help="Max examples for Mistral evaluation (default: 30, stratified)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print_header("Loading dataset")
    df = load_dataset(args.labels)
    print(f"  Total templates : {len(df)}")
    print(f"  Label counts    : {dict(df['label'].value_counts())}")

    df_train, df_test = split_dataset(df)
    test_labels = sorted(df_test["label"].unique().tolist())
    print(f"  Train: {len(df_train)}  |  Test: {len(df_test)}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv_results = {}
    if args.crossval:
        print_header("5-fold Cross-Validation")
        cv_results = run_crossval(df)
        print()
        for name, res in cv_results.items():
            print(
                f"  {name:<22}  macro F1 = {res['mean']:.4f} ± {res['std']:.4f}"
                f"  95% CI=[{res['ci95'][0]:.3f}, {res['ci95'][1]:.3f}]"
            )

    # ── Train and evaluate all classifiers ────────────────────────────────────
    print_header("Loading pre-trained models")

    MODEL_FILES = {
        "Naive Bayes (CNB)": "naive_bayes.joblib",
        "Random Forest": "random_forest.joblib",
        "LogReg (baseline)": "logreg.joblib",
        "LinearSVC": "linearsvc.joblib",
    }
    # Initialise accumulators before any model evaluation block
    rows = []
    all_preds = {}
    best_f1 = 0.0
    best_name = None
    llm_details = []

    for name, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if not path.exists():
            print(f"  {YELLOW}Missing: {path} — run train_model.py first{RESET}")
            continue

        pipe = joblib.load(path)
        t0 = time.time()
        y_pred = pipe.predict(df_test["template"].tolist()).tolist()
        elapsed = time.time() - t0

        row = metrics_row(name, df_test["label"].tolist(), y_pred)
        row["Time (s)"] = round(elapsed, 2)
        row["_best"] = False
        rows.append(row)
        all_preds[name] = y_pred

        if row["Macro F1"] > best_f1:
            best_f1 = row["Macro F1"]
            best_name = name

        print(f"  ✓ {name:<22}  macro F1={row['Macro F1']:.3f}  ({elapsed:.1f}s)")

    # Mark best model row for bold display
    for r in rows:
        r["_best"] = r["Model"] == best_name

    # ── LLM evaluation ────────────────────────────────────────────────────────
    llm_details = []
    if args.llm:
        print_header(f"LLM: Phi-3.5-Mini (zero-shot) — {args.llm_limit} examples")
        print(
            "  Make sure Ollama is running: ollama serve (It should be pre-installed!)"
        )
        try:
            llm = LLMEvaluator()
            df_llm = df_test.head(args.llm_limit)

            t0 = time.time()
            llm_labels, llm_confs, llm_times = llm.predict(
                df_llm["template"].tolist(), limit=args.llm_limit
            )
            total_time = time.time() - t0

            print(
                f"  Total: {total_time:.0f}s | "
                f"mean={np.mean(llm_times):.0f}s/item | "
                f"p95={np.percentile(llm_times, 95):.0f}s/item"
            )

            # Filter out "unknown" predictions before computing metrics
            y_true_llm = df_llm["label"].tolist()
            valid_pairs = [
                (t, p) for t, p in zip(y_true_llm, llm_labels) if p in test_labels
            ]
            n_unknown = len(llm_labels) - len(valid_pairs)

            if n_unknown:
                print(
                    f"  {YELLOW}Excluded {n_unknown} 'unknown' responses"
                    f" from metrics{RESET}"
                )

            if valid_pairs:
                yt, yp = zip(*valid_pairs)
                row = metrics_row(llm.name, list(yt), list(yp))
                row["Time (s)"] = round(float(np.mean(llm_times)), 1)
                row["_best"] = False
                rows.append(row)

                present = sorted(set(yt) | set(yp))
                print_detail_report(llm.name, list(yt), list(yp), present)

            # Store details for report qualitative examples table
            llm_details = [
                {
                    "template": df_llm.iloc[i]["template"],
                    "true_label": y_true_llm[i],
                    "pred_label": llm_labels[i],
                    "confidence": llm_confs[i],
                    "inference_sec": llm_times[i],
                }
                for i in range(len(llm_labels))
            ]

        except Exception as e:
            print(f"  {RED}LLM unavailable: {e}{RESET}")
            print("  Set NO_PROXY=localhost,127.0.0.1 and run: ollama serve")

    # ── Mistral-7B evaluation ────────────────────────────────────────────────────
    if args.mistral:
        print_header(
            f"Mistral-7B (zero-shot) — {args.mistral_limit} examples (stratified)"
        )
        print("  Make sure Ollama is running: ollama serve")
        try:
            mistral = MistralEvaluator()

            # Stratified sample — equal examples per class

            n_per_class = max(1, args.mistral_limit // len(test_labels))
            frames = []
            for lbl in test_labels:
                lbl_df = df_test[df_test["label"] == lbl]
                n = min(len(lbl_df), n_per_class)
                frames.append(lbl_df.sample(n, random_state=RANDOM_STATE))
            df_mis = pd.concat(frames).reset_index(drop=True)
            print(f"  Sample: {dict(df_mis['label'].value_counts())}")

            t0 = time.time()
            mis_labels, mis_confs, mis_times = mistral.predict(
                df_mis["template"].tolist(), limit=len(df_mis)
            )
            total_time = time.time() - t0

            print(
                f"  Total: {total_time:.0f}s | "
                f"mean={np.mean(mis_times):.0f}s/item | "
                f"p95={np.percentile(mis_times, 95):.0f}s/item"
            )

            y_true_mis = df_mis["label"].tolist()
            valid_pairs = [
                (t, p) for t, p in zip(y_true_mis, mis_labels) if p in test_labels
            ]
            n_unknown = len(mis_labels) - len(valid_pairs)

            if n_unknown:
                print(f"  {YELLOW}Excluded {n_unknown} 'unknown' responses{RESET}")

            if valid_pairs:
                yt, yp = zip(*valid_pairs)
                row = metrics_row(mistral.name, list(yt), list(yp))
                row["Time (s)"] = round(float(np.mean(mis_times)), 1)
                row["_best"] = False
                rows.append(row)
                if row["Macro F1"] > best_f1:
                    best_f1 = row["Macro F1"]
                    best_name = mistral.name
                    for r in rows:
                        r["_best"] = r["Model"] == best_name
                print_detail_report(
                    mistral.name, list(yt), list(yp), sorted(set(yt) | set(yp))
                )

        except Exception as e:
            print(f"  {RED}Mistral unavailable: {e}{RESET}")
            print("  Run: ollama serve")

    # ── LogBERT evaluation ────────────────────────────────────────────────────
    if args.logbert:
        print_header("LogBERT (fine-tuned BERT)")
        try:
            logbert = LogBERTEvaluator()
            t0 = time.time()
            lb_labels, lb_confs, lb_times = logbert.predict(
                df_test["template"].tolist()  # evaluate on full test set
            )
            print(
                f"  Total: {time.time()-t0:.0f}s | "
                f"mean={np.mean(lb_times):.3f}s/item"
            )

            valid = [
                (t, p)
                for t, p in zip(df_test["label"].tolist(), lb_labels)
                if p in test_labels
            ]
            if valid:
                yt, yp = zip(*valid)
                row = metrics_row(logbert.name, list(yt), list(yp))
                row["Time (s)"] = round(float(np.mean(lb_times)), 3)
                row["_best"] = False
                rows.append(row)
                # Update best model if LogBERT beats current best
                if row["Macro F1"] > best_f1:
                    best_f1 = row["Macro F1"]
                    best_name = logbert.name
                    for r in rows:
                        r["_best"] = r["Model"] == best_name
                print_detail_report(
                    logbert.name, list(yt), list(yp), sorted(set(yt) | set(yp))
                )
        except Exception as e:
            print(f"  {RED}LogBERT unavailable: {e}{RESET}")
            print("  Fine-tune on Colab first, save to models/logbert_anomalens_final/")

    # ── Print final comparison table ──────────────────────────────────────────
    print_header("Final comparison")
    print_comparison_table(rows)  # pops _best key from each row

    if best_name:
        print(
            f"\n  {GREEN}Best model: {BOLD}{best_name}{RESET}"
            f"{GREEN}  macro F1 = {best_f1:.3f}{RESET}"
        )

    # Detailed per-class report for the best classical classifier
    if best_name and best_name in all_preds:
        print_detail_report(
            best_name,
            df_test["label"].tolist(),
            all_preds[best_name],
            test_labels,
        )

    # ── Save everything ───────────────────────────────────────────────────────
    clean_rows = [{k: v for k, v in r.items() if k != "_best"} for r in rows]
    save_results(clean_rows, cv_results, llm_details, run_id, args)

    print(f"\n{BOLD}Done. Run ID: {run_id}{RESET}\n")


if __name__ == "__main__":
    main()
