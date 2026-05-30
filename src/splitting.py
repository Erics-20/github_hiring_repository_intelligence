"""
Stage 4 – Stratified Train / Validation / Test Split
Splits the LLM-labeled dataset into three stratified sets (70 / 15 / 15)
and saves them to data/splits/.

Input:  data/labeled/llm_labeled.csv
Output:
    data/splits/train.csv   (70 %)
    data/splits/val.csv     (15 %)
    data/splits/test.csv    (15 %)
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ── Paths ──────────────────────────────────────────────────────────────────────
LABELED_PATH = Path("data/labeled/llm_labeled.csv")
SPLITS_DIR   = Path("data/splits")

TARGET_COL   = "llm_label"
RANDOM_STATE = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_split_stats(name: str, df: pd.DataFrame, total: int) -> None:
    pct_of_total = len(df) / total * 100
    print(f"\n{'─' * 52}")
    print(f"  {name}  —  {len(df):,} rows  ({pct_of_total:.1f}% of full dataset)")
    print(f"{'─' * 52}")
    counts = df[TARGET_COL].value_counts()
    pcts   = df[TARGET_COL].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "%": pcts.round(1)})
    print(summary.to_string())


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    labeled_path: Path = LABELED_PATH,
    splits_dir:   Path = SPLITS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the two-step stratified split and persist the three CSV files.

    Step 1 — isolate 15 % as the test set  (stratified on TARGET_COL).
    Step 2 — split the remaining 85 % into train (70 %) and val (15 %)
             using val_size = 15/85 to recover the exact original ratios.

    Returns (train_df, val_df, test_df).
    """
    print(f"[splitting] Reading labeled data from: {labeled_path}")
    df = pd.read_csv(labeled_path)
    total = len(df)
    print(f"[splitting] Loaded {total:,} rows.")
    print(f"\n[splitting] Full dataset — class distribution:")
    print(df[TARGET_COL].value_counts().to_string())

    # ── Step 1: isolate Test (15 %) ───────────────────────────────────────────
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )

    # ── Step 2: split remaining 85 % into Train (70 %) and Val (15 %) ────────
    # val_size within the 85 % slice that recovers exactly 15 % of the total
    val_ratio_within_trainval = 0.15 / 0.85

    train, val = train_test_split(
        train_val,
        test_size=val_ratio_within_trainval,
        stratify=train_val[TARGET_COL],
        random_state=RANDOM_STATE,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    splits_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir   / "val.csv",   index=False)
    test.to_csv(splits_dir  / "test.csv",  index=False)

    # ── Verification report ───────────────────────────────────────────────────
    print(f"\n{'=' * 52}")
    print("  SPLIT VERIFICATION REPORT")
    print(f"{'=' * 52}")
    _print_split_stats("TRAIN", train, total)
    _print_split_stats("VAL  ", val,   total)
    _print_split_stats("TEST ", test,  total)

    reconstructed = len(train) + len(val) + len(test)
    print(f"\n{'=' * 52}")
    print(f"  Total rows accounted for: {reconstructed:,} / {total:,}")
    print(f"  Splits saved to: {splits_dir.resolve()}")
    print(f"{'=' * 52}\n")

    assert reconstructed == total, "Row count mismatch — rows lost in split!"

    return train, val, test


if __name__ == "__main__":
    run()
