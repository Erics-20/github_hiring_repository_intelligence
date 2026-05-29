"""
Stage 2 – Text Representation
Reads raw collected data and produces a hybrid text representation for each
repository, optimized for LLM weak-labeling and BERT fine-tuning.

Output: data/processed/text_representations.csv
"""

import pandas as pd
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATH = Path("data/raw/collected.csv")
PROCESSED_PATH = Path("data/processed/text_representations.csv")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_str(value, default: str = "None") -> str:
    """Return a stripped string, substituting default when value is null."""
    if pd.isna(value) or str(value).strip() == "":
        return default
    return str(value).strip()


def _clean_bool(value) -> str:
    """Normalise boolean-like values to the string 'True' or 'False'."""
    if pd.isna(value):
        return "False"
    if isinstance(value, bool):
        return str(value)
    normalized = str(value).strip().lower()
    return "True" if normalized in ("true", "1", "yes") else "False"


def build_text_representation(row: pd.Series) -> str:
    """
    Construct the hybrid text representation for a single repository row.

    Template:
        [METRICS] ... [DESCRIPTION] ... [TOPICS] ...
    """
    language    = _clean_str(row.get("language"),    default="None")
    stars       = int(row.get("stargazers_count", 0) or 0)
    forks       = int(row.get("forks_count", 0)      or 0)
    issues      = int(row.get("open_issues_count", 0) or 0)
    size        = int(row.get("size", 0)             or 0)
    has_wiki    = _clean_bool(row.get("has_wiki"))
    has_pages   = _clean_bool(row.get("has_pages"))
    license_    = _clean_str(row.get("license"),     default="No license")
    archived    = _clean_bool(row.get("archived"))
    description = _clean_str(row.get("description"), default="No description provided.")
    topics      = _clean_str(row.get("topics"),      default="None")

    return (
        f"[METRICS] This repository is written in {language}. "
        f"It has {stars} stars, {forks} forks, and {issues} open issues. "
        f"Size: {size} KB. "
        f"Wiki enabled: {has_wiki}. "
        f"Pages enabled: {has_pages}. "
        f"License: {license_}. "
        f"Archived status: {archived}. "
        f"[DESCRIPTION] {description}"
        f"[TOPICS] {topics}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def run(raw_path: Path = RAW_PATH, output_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load raw data, apply the text representation, and persist the result.

    Returns the enriched DataFrame.
    """
    print(f"[preprocessing] Reading raw data from: {raw_path}")
    df = pd.read_csv(raw_path, dtype=str)          # read everything as str first
    print(f"[preprocessing] Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # Numeric columns need proper types for arithmetic safety
    for col in ("stargazers_count", "forks_count", "open_issues_count", "size"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["text_representation"] = df.apply(build_text_representation, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocessing] Saved enriched data to: {output_path}")
    print(f"[preprocessing] Sample text representation (row 0):\n")
    print(df["text_representation"].iloc[0])

    return df


if __name__ == "__main__":
    run()
