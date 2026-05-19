"""
train_model.py — train all classifiers and save to models/.

Run this once after updating labels.csv.
The best model (LinearSVC) is saved as model.joblib to use as Default model
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

INPUT_FILE = "data/labels/labels.csv"
MODELS_DIR = Path("models")
PROD_MODEL = "model.joblib"  # default model used by predict.py
RANDOM_STATE = 42

CLASSIFIERS = {
    "naive_bayes": ComplementNB(),
    "random_forest": RandomForestClassifier(
        100, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    ),
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "linearsvc": LinearSVC(max_iter=2000, class_weight="balanced"),
}

# Best model - saved as default model into model.joblib
BEST_MODEL = "linearsvc"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["template"] = df["template"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df = df[df["template"] != ""]
    df = df[df["label"].isin(["noise", "important", "error", "security_noise"])]
    return df


def main():
    df = load_dataset(INPUT_FILE)
    print(f"Dataset: {len(df)} templates")
    print(dict(df["label"].value_counts()))

    X_train, X_test, y_train, y_test = train_test_split(
        df["template"],
        df["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    MODELS_DIR.mkdir(exist_ok=True)

    for name, clf in CLASSIFIERS.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                ("clf", clf),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print(classification_report(y_test, y_pred, zero_division=0))

        # Save each model
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved → {model_path}")

        # Save best model as default model
        if name == BEST_MODEL:
            joblib.dump(pipe, PROD_MODEL)
            print(f"Saved → {PROD_MODEL}  (default)")

    print(f"\nDefault model: {BEST_MODEL}")
    print(f"Saved to: {PROD_MODEL}")
    print(f"Classes: {CLASSIFIERS[BEST_MODEL].__class__.__name__}")


if __name__ == "__main__":
    main()
