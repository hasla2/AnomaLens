"""
hdfs_benchmark.py — TF-IDF + LinearSVC on HDFS public benchmark.

Compares AnomaLens classification approach against published results
from DeepLog and LogBERT on the standard HDFS log anomaly detection task.

Dataset: HDFS_v1 (loghub) — preprocessed version
    Event_traces.csv      — session sequences + labels (Success/Fail)
    HDFS.log_templates.csv — event ID → template text mapping

Usage:
    python hdfs_benchmark.py --data data/hdfs/preprocessed
    python hdfs_benchmark.py --data data/hdfs/preprocessed --sample 50000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42

# ── Published baselines ────────────────────────────────────────────────────────
PUBLISHED = {
    "DeepLog [Du et al., 2017]": {"P": 0.9615, "R": 0.9375, "F1": 0.9493},
    "LogBERT [Guo et al., 2021]": {"P": 0.9827, "R": 0.9790, "F1": 0.9808},
}


def load_templates(data_dir: Path) -> dict:
    """Load event ID → template text mapping."""
    df = pd.read_csv(data_dir / "HDFS.log_templates.csv")
    print(f"  {len(df)} unique event templates")
    return dict(zip(df["EventId"].astype(str), df["EventTemplate"].astype(str)))


def events_to_text(seq_str: str, event_map: dict) -> str:
    """Convert '[E5,E22,E11]' → 'template5 template22 template11'."""
    seq_str = str(seq_str).strip("[]")
    events = [e.strip() for e in seq_str.split(",")]
    return " ".join(event_map.get(e, e) for e in events if e)


def load_dataset(data_dir: Path, event_map: dict) -> pd.DataFrame:
    """Load Event_traces.csv and convert to (document, label) format."""
    print("Loading event traces...")
    df = pd.read_csv(data_dir / "Event_traces.csv")
    print(f"  {len(df):,} sessions loaded")
    print(f"  Columns: {list(df.columns)}")

    # Label: Success=0 (Normal), Fail=1 (Anomaly)
    df["label"] = (df["Label"].str.strip() == "Fail").astype(int)
    print(f"  Normal:  {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"  Anomaly: {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")

    print("Converting event sequences to template text...")
    df["document"] = df["Features"].apply(lambda x: events_to_text(x, event_map))

    print(f"  Example: {df['document'].iloc[0][:120]}")
    return df[["document", "label"]].dropna()


def evaluate(name: str, pipe, X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate one pipeline."""
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return {
        "Model": name,
        "P": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "R": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


def main():
    p = argparse.ArgumentParser(
        description="AnomaLens HDFS benchmark vs DeepLog/LogBERT"
    )
    p.add_argument(
        "--data",
        default="data/hdfs/preprocessed",
        help="Path to preprocessed HDFS directory",
    )
    p.add_argument(
        "--sample", type=int, default=None, help="Use only N sessions for quick test"
    )
    args = p.parse_args()

    data_dir = Path(args.data)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading HDFS dataset")
    print("=" * 60)

    event_map = load_templates(data_dir)
    df = load_dataset(data_dir, event_map)

    if args.sample:
        df = df.sample(args.sample, random_state=RANDOM_STATE)
        print(f"\nSampled: {len(df):,} sessions")

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["document"],
        df["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Models ────────────────────────────────────────────────────────────────
    MODELS = {
        "Naive Bayes (CNB)": ComplementNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            100, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "LinearSVC": LinearSVC(max_iter=2000, class_weight="balanced"),
    }

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"{'Model':<35} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 60)

    results = []
    for name, clf in MODELS.items():
        pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=30_000,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", clf),
            ]
        )
        r = evaluate(name, pipe, X_train, X_test, y_train, y_test)
        results.append(r)
        print(f"  {name:<33} {r['P']:>7.4f} {r['R']:>7.4f} {r['F1']:>7.4f}")

    print("-" * 60)
    print("  Published baselines:")
    for name, r in PUBLISHED.items():
        results.append({"Model": name, **r})
        print(f"  {name:<33} {r['P']:>7.4f} {r['R']:>7.4f} {r['F1']:>7.4f}")

    print("=" * 60)

    # ── Best model detail ─────────────────────────────────────────────────────
    best = max([r for r in results if r["Model"] in MODELS], key=lambda x: x["F1"])
    print(f"\nBest model: {best['Model']}  F1={best['F1']:.4f}")

    # Retrain best for detailed report
    best_pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True
                ),
            ),
            ("clf", MODELS[best["Model"]]),
        ]
    )
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    print(f"\nDetailed report ({best['Model']}):")
    print(
        classification_report(
            y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0
        )
    )

    # ── Save results ──────────────────────────────────────────────────────────
    out = Path("data/results/hdfs_benchmark.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
