"""
Stage 3 – Weak Labeling with DeepSeek
Classifies each repository's engineering maturity using zero-shot prompting
via the DeepSeek API (OpenAI-compatible client).

Input:  data/processed/text_representations.csv  (text_representation column)
Output: data/labeled/llm_labeled.csv             (adds llm_label column)
"""

import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

PROCESSED_PATH = Path("data/processed/text_representations.csv")
LABELED_PATH   = Path("data/labeled/llm_labeled.csv")

MODEL          = "deepseek-chat"
BASE_URL       = "https://api.deepseek.com"
MAX_RETRIES    = 3
RETRY_DELAY    = 10   # seconds between retries on rate-limit / transient errors

VALID_LABELS = {
    "Low-value / Template",
    "Intern / Junior-level",
    "Senior-level / Lead / Architect-level",
}

SYSTEM_PROMPT = (
    "You are an expert Software Engineering Evaluator. "
    "Your task is to read the text representation of a GitHub repository's metadata "
    "and classify its engineering maturity strictly into ONE of the following three categories:\n"
    "1. 'Low-value / Template'\n"
    "2. 'Intern / Junior-level'\n"
    "3. 'Senior-level / Lead / Architect-level'\n\n"
    "Evaluate based on the following dimensions:\n"
    "  • Collaboration & Activity (contributors, forks, issues, recent updates).\n"
    "  • Documentation (presence of wiki, pages, and clear description).\n"
    "  • Governance (presence of an open-source license).\n"
    "  • Originality & Complexity (is it a fork? what is its size and technological purpose?).\n\n"
    "DO NOT provide any explanations, greetings, or additional text. "
    "Your response must be EXACTLY the name of the category and nothing else."
)


# ── API client ─────────────────────────────────────────────────────────────────
def build_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY not found. Add it to your .env file."
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL)


# ── Single-row labeling ────────────────────────────────────────────────────────
def classify_repository(client: OpenAI, text_representation: str) -> str:
    """
    Send one classification request to DeepSeek.
    Returns the label string, or an error sentinel on persistent failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text_representation},
                ],
                temperature=0,      # deterministic for labeling tasks
                max_tokens=20,      # label is always short
            )
            raw = response.choices[0].message.content
            label = raw.strip().strip("`").strip("'").strip('"')
            return label

        except RateLimitError:
            if attempt < MAX_RETRIES:
                tqdm.write(f"  [rate limit] Attempt {attempt}/{MAX_RETRIES} — waiting {RETRY_DELAY}s…")
                time.sleep(RETRY_DELAY * attempt)   # exponential back-off
            else:
                tqdm.write("  [rate limit] Max retries reached. Storing error sentinel.")
                return "ERROR_RATE_LIMIT"

        except (APIConnectionError, APITimeoutError) as exc:
            if attempt < MAX_RETRIES:
                tqdm.write(f"  [connection error] {exc} — retrying in {RETRY_DELAY}s…")
                time.sleep(RETRY_DELAY)
            else:
                tqdm.write(f"  [connection error] Max retries reached: {exc}")
                return "ERROR_CONNECTION"

        except Exception as exc:
            tqdm.write(f"  [unexpected error] {exc}")
            return "ERROR_UNEXPECTED"

    return "ERROR_UNKNOWN"


# ── Main ───────────────────────────────────────────────────────────────────────
def run(
    input_path: Path  = PROCESSED_PATH,
    output_path: Path = LABELED_PATH,
) -> pd.DataFrame:
    print(f"[llm_labeling] Reading processed data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[llm_labeling] Loaded {len(df):,} rows.")

    if "text_representation" not in df.columns:
        raise ValueError("Column 'text_representation' not found. Run preprocessing.py first.")

    # Resume support: skip rows already labeled in a previous run
    if "llm_label" not in df.columns:
        df["llm_label"] = None

    pending_mask = df["llm_label"].isna()
    pending_count = pending_mask.sum()
    print(f"[llm_labeling] Rows to label: {pending_count:,}  (already labeled: {(~pending_mask).sum():,})")

    if pending_count == 0:
        print("[llm_labeling] Nothing to do. Output is already fully labeled.")
        return df

    client = build_client()

    for idx in tqdm(df.index[pending_mask], desc="Labeling repositories", unit="repo"):
        text = df.at[idx, "text_representation"]
        label = classify_repository(client, str(text))

        if label not in VALID_LABELS and not label.startswith("ERROR"):
            tqdm.write(f"  [warn] Unexpected label at row {idx}: '{label}' — storing as-is.")

        df.at[idx, "llm_label"] = label

        # Checkpoint: save after every row so progress survives interruptions
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n[llm_labeling] Done. Output saved to: {output_path}")
    print("\n── Label distribution ──────────────────────────────────────")
    print(df["llm_label"].value_counts().to_string())

    errors = df["llm_label"].str.startswith("ERROR", na=False).sum()
    if errors:
        print(f"\n[warn] {errors} rows could not be labeled (API errors). Review and re-run.")

    return df


if __name__ == "__main__":
    run()
