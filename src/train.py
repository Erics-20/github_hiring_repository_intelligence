"""
Stage 5 – DistilBERT Fine-Tuning for Repository Maturity Classification
========================================================================
Run in Google Colab
-------------------
# 1. Mount Drive and navigate to the project root
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/<your_path>/github_hiring_repository_intelligence

# 2. Install dependencies
!pip install transformers datasets accelerate scikit-learn -q

# 3. Run
!python src/train.py

Input:
    data/splits/train.csv
    data/splits/val.csv

Output:
    models/checkpoints/             (per-epoch checkpoints)
    models/trained_models/distilbert_repo_classifier/  (final model + tokenizer)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 256           # truncates long descriptions; fits Colab T4 VRAM
TEXT_COL    = "text_representation"
LABEL_COL   = "llm_label"

TRAIN_PATH  = Path("data/splits/train.csv")
VAL_PATH    = Path("data/splits/val.csv")
CKPT_DIR    = Path("models/checkpoints")
FINAL_DIR   = Path("models/trained_models/distilbert_repo_classifier")

# Explicit mapping — order determines the integer id fed to CrossEntropyLoss
LABEL2ID = {
    "Low-value / Template":                  0,
    "Intern / Junior-level":                 1,
    "Senior-level / Lead / Architect-level": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# ── 1. Data Loading ────────────────────────────────────────────────────────────
def load_splits(
    train_path: Path = TRAIN_PATH,
    val_path:   Path = VAL_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    for df in (train_df, val_df):
        df["label"] = df[LABEL_COL].map(LABEL2ID)
        if df["label"].isna().any():
            unknown = df.loc[df["label"].isna(), LABEL_COL].unique()
            raise ValueError(f"Unmapped labels found: {unknown}")

    print(f"[train] Train rows : {len(train_df):,}")
    print(f"[train] Val rows   : {len(val_df):,}")
    print(f"\n[train] Label mapping: {LABEL2ID}")
    return train_df, val_df


# ── 2. Class Weights ───────────────────────────────────────────────────────────
def get_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    classes  = np.array(sorted(LABEL2ID.values()))
    weights  = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values,
    )
    tensor   = torch.tensor(weights, dtype=torch.float)
    print(f"\n[train] Class weights:")
    for label_id, w in zip(classes, weights):
        print(f"         {ID2LABEL[label_id]:<45} → {w:.4f}")
    return tensor


# ── 3. Tokenization ────────────────────────────────────────────────────────────
def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: DistilBertTokenizerFast,
) -> Dataset:
    hf_dataset = Dataset.from_pandas(
        df[[TEXT_COL, "label"]].reset_index(drop=True)
    )

    def _tokenize(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    return hf_dataset.map(_tokenize, batched=True)


# ── 4. Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


# ── 5. Custom Trainer (weighted loss) ─────────────────────────────────────────
class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies per-class weights to CrossEntropyLoss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── 6. Training Arguments ──────────────────────────────────────────────────────
def build_training_args() -> TrainingArguments:
    use_fp16 = torch.cuda.is_available()
    # Use batch 16 on GPU (Colab T4/A100), 8 on CPU
    batch_size = 16 if torch.cuda.is_available() else 8

    print(f"\n[train] Device : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"[train] fp16   : {use_fp16}")
    print(f"[train] Batch  : {batch_size}")

    return TrainingArguments(
        output_dir                  = str(CKPT_DIR),
        num_train_epochs            = 3,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size * 2,
        learning_rate               = 2e-5,
        weight_decay                = 0.01,
        eval_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        fp16                        = use_fp16,
        logging_dir                 = str(CKPT_DIR / "logs"),
        logging_strategy            = "epoch",
        report_to                   = "none",   # disable wandb / tensorboard
        save_total_limit            = 2,        # keep only the 2 best checkpoints
    )


# ── 7. Main ────────────────────────────────────────────────────────────────────
def run():
    # ── Load data ──────────────────────────────────────────────────────────────
    train_df, val_df = load_splits()

    # ── Class weights ──────────────────────────────────────────────────────────
    class_weights = get_class_weights(train_df)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    print(f"\n[train] Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = tokenize_dataset(train_df, tokenizer)
    val_dataset   = tokenize_dataset(val_df,   tokenizer)

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"[train] Loading model : {MODEL_NAME}  ({NUM_LABELS} labels)")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels  = NUM_LABELS,
        id2label    = ID2LABEL,
        label2id    = LABEL2ID,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = WeightedLossTrainer(
        class_weights   = class_weights,
        model           = model,
        args            = build_training_args(),
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        processing_class       = tokenizer,
        compute_metrics = compute_metrics,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\n[train] Starting training …\n")
    trainer.train()

    # ── Evaluate best model on validation set ──────────────────────────────────
    print("\n[train] Final evaluation on validation set:")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"         {k:<30} {v}")

    # ── Save final model + tokenizer ───────────────────────────────────────────
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))
    print(f"\n[train] Model saved to: {FINAL_DIR.resolve()}")

    # Save label mapping alongside the model for inference
    import json
    mapping = {"label2id": LABEL2ID, "id2label": ID2LABEL}
    with open(FINAL_DIR / "label_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[train] Label mapping saved to: {FINAL_DIR / 'label_mapping.json'}")


if __name__ == "__main__":
    run()
