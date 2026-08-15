from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import joblib

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_PATH = DATA_DIR / "ag_news_train.csv"
TEST_PATH = DATA_DIR / "ag_news_test.csv"
LABEL_ORDER = ["World", "Sports", "Business", "Sci_Tech"]


def preprocess_text(text: str) -> str:
    """Limpia el texto antes de vectorizarlo con TF-IDF."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_columns = {"text", "label"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas en {path.name}: {sorted(missing_columns)}")
    if df[["text", "label"]].isna().sum().sum() > 0:
        raise ValueError(f"Hay valores nulos en text/label dentro de {path.name}")
    return df


def build_pipeline(max_features: int, ngram_max: int, c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    stop_words="english",
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Matriz de confusión - TF-IDF + Regresión Logística")
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_and_evaluate(max_features: int, ngram_max: int, c_value: float) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    test_df = load_split(TEST_PATH)

    x_train = train_df["text"]
    y_train = train_df["label"]
    x_test = test_df["text"]
    y_test = test_df["label"]

    pipeline = build_pipeline(max_features=max_features, ngram_max=ngram_max, c_value=c_value)

    # Importante: fit se ejecuta solo sobre train. Test queda completamente no visto.
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(y_test, y_pred, labels=LABEL_ORDER, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    vectorizer = pipeline.named_steps["tfidf"]
    vocabulary_size = len(vectorizer.vocabulary_)

    metrics = {
        "dataset": "AG News",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "model": "LogisticRegression",
        "vectorizer": "TfidfVectorizer",
        "max_features": max_features,
        "ngram_range": [1, ngram_max],
        "min_df": 2,
        "stop_words": "english",
        "sublinear_tf": True,
        "vocabulary_size": int(vocabulary_size),
        "accuracy": float(accuracy),
        "classification_report": report,
    }

    (OUTPUT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_confusion_matrix(y_test, y_pred, OUTPUT_DIR / "confusion_matrix.png")
    joblib.dump(pipeline, MODEL_DIR / "tfidf_logistic_regression.joblib")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un clasificador supervisado TF-IDF para AG News."
    )
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--c-value", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_and_evaluate(
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        c_value=args.c_value,
    )
    print("Entrenamiento finalizado correctamente.")
    print(f"Train: {result['train_rows']} documentos")
    print(f"Test: {result['test_rows']} documentos")
    print(f"Vocabulario TF-IDF: {result['vocabulary_size']} términos")
    print(f"Accuracy test: {result['accuracy']:.4f}")
