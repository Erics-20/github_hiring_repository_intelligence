"""
Stage 6 – Evaluation and Error Analysis
========================================
Run in Google Colab
-------------------
# 1. Mount Drive and navigate to the project root
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/<your_path>/github_hiring_repository_intelligence

# 2. Install dependencies
!pip install transformers scikit-learn seaborn matplotlib pandas -q

# 3. Run
!python src/evaluation.py

Input:
    data/splits/test.csv
    models/trained_models/distilbert_repo_classifier/

Output:
    output/metrics/classification_report.json
    output/metrics/evaluation_summary.txt
    output/figures/confusion_matrix.png
    output/tables/misclassified_examples.csv
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ── Constants ──────────────────────────────────────────────────────────────────
TEXT_COL   = "text_representation"
LABEL_COL  = "llm_label"
MAX_LENGTH = 256
BATCH_SIZE = 32

TEST_PATH  = Path("data/splits/test.csv")
MODEL_DIR  = Path("models/trained_models/distilbert_repo_classifier")

FIG_DIR     = Path("output/figures")
TABLE_DIR   = Path("output/tables")
METRICS_DIR = Path("output/metrics")

# Must match the mapping used in train.py exactly
LABEL2ID = {
    "Low-value / Template":                  0,
    "Intern / Junior-level":                 1,
    "Senior-level / Lead / Architect-level": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ── 1. Environment Setup ───────────────────────────────────────────────────────
def setup_dirs() -> None:
    for d in (FIG_DIR, TABLE_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print("[eval] Output directories ready.")


# ── 2. Data Loading ────────────────────────────────────────────────────────────
def load_test_data(path: Path = TEST_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["true_id"] = df[LABEL_COL].map(LABEL2ID)
    if df["true_id"].isna().any():
        unknown = df.loc[df["true_id"].isna(), LABEL_COL].unique()
        raise ValueError(f"Unmapped labels in test set: {unknown}")
    print(f"[eval] Test rows loaded : {len(df):,}")
    print(f"[eval] Label distribution:\n{df[LABEL_COL].value_counts().to_string()}\n")
    return df


# ── 3. Inference ───────────────────────────────────────────────────────────────
def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_inference(df: pd.DataFrame) -> np.ndarray:
    print(f"[eval] Loading model from : {MODEL_DIR.resolve()}")
    classifier = pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        device=0 if _cuda_available() else -1,
        batch_size=BATCH_SIZE,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    texts = df[TEXT_COL].tolist()
    print(f"[eval] Running inference on {len(texts):,} examples …")
    results = classifier(texts)

    pred_ids = np.array([LABEL2ID[r["label"]] for r in results], dtype=int)
    print(f"[eval] Inference complete.\n")
    return pred_ids


# ── 4. Metrics Calculation & Storage ──────────────────────────────────────────
def compute_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

    accuracy                  = accuracy_score(y_true, y_pred)
    precision, recall, f1, _  = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    )

    print("[eval] Classification Report:")
    print(report_str)

    # ── classification_report.json ────────────────────────────────────────────
    report_path = METRICS_DIR / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[eval] Saved : {report_path}")

    # ── evaluation_summary.txt ────────────────────────────────────────────────
    summary_path = METRICS_DIR / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Evaluation Summary (macro averages) ===\n\n")
        f.write(f"Accuracy  : {accuracy:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write(f"F1-score  : {f1:.4f}\n\n")
        f.write("=== Full Classification Report ===\n\n")
        f.write(report_str)
    print(f"[eval] Saved : {summary_path}")


# ── 5. Confusion Matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix – DistilBERT Repository Classifier", fontsize=13)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig_path = FIG_DIR / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Saved : {fig_path}")


# ── 6. Error Analysis ──────────────────────────────────────────────────────────
def save_misclassified(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    mask     = y_true != y_pred
    n_errors = int(mask.sum())
    print(f"\n[eval] Misclassified examples : {n_errors} / {len(y_true)}"
          f"  ({n_errors / len(y_true) * 100:.1f}%)")

    error_df = df[mask].copy().reset_index(drop=True)
    error_df["true_label"]      = [ID2LABEL[i] for i in y_true[mask]]
    error_df["predicted_label"] = [ID2LABEL[i] for i in y_pred[mask]]

    out_cols = ["full_name", TEXT_COL, "true_label", "predicted_label"]
    csv_path = TABLE_DIR / "misclassified_examples.csv"
    error_df[out_cols].to_csv(csv_path, index=False)
    print(f"[eval] Saved : {csv_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run() -> None:
    setup_dirs()

    df     = load_test_data()
    y_true = df["true_id"].values.astype(int)
    y_pred = run_inference(df)

    compute_and_save_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    save_misclassified(df, y_true, y_pred)

    print("\n[eval] Stage 6 complete. All outputs written to output/")


if __name__ == "__main__":
    run()
