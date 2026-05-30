# GitHub Hiring Repository Intelligence

An end-to-end NLP pipeline that automatically evaluates GitHub repositories submitted by job candidates and classifies them by engineering maturity — helping recruiters focus manual review on repositories that carry real hiring signal.

---

## What does the project do?

The pipeline collects public GitHub repositories via the GitHub Search API, builds a structured text representation of each one from its metadata, uses **DeepSeek** to weakly label them across three engineering-maturity tiers, and fine-tunes a **DistilBERT** classifier on those labels. The trained model is served through a **Streamlit** dashboard where recruiters can filter repositories, inspect metadata, and run live inference.

---

## Which track was selected?

**Applied NLP / Weak Supervision track.**  
The project combines GitHub API data engineering, LLM-based weak labeling (zero-shot prompting with DeepSeek), and BERT fine-tuning into a single hiring-intelligence system. The focus is on using imperfect, automatically generated labels to train a model that can scale repository screening without manual annotation.

---

## What repositories were analyzed?

**450 public GitHub repositories** were collected across three stratified buckets using the GitHub Search API:

| Bucket | Query Strategy | Count |
|---|---|---|
| Low-value / Template | 0–2 stars, no description, "template" in name | 296 |
| Intern / Junior-level | Keywords: *homework*, *bootcamp*, *learning*, *assignment* | 22 |
| Senior / Lead / Architect | ≥ 50 stars, recently active, non-trivial topics | 132 |

Each bucket was capped at 150 requests to stay within API rate limits. Repos were deduplicated and filtered to exclude private or archived repositories.

---

## Which GitHub signals were used?

Each repository was described by the following metadata fields, all extracted via the GitHub REST API v3:

| Signal | Type | Rationale |
|---|---|---|
| `stargazers_count` | Numeric | Community validation proxy |
| `forks_count` | Numeric | Reuse and credibility indicator |
| `open_issues_count` | Numeric | Active maintenance signal |
| `size` (KB) | Numeric | Rough codebase complexity proxy |
| `language` | Categorical | Primary programming language |
| `topics` | Text list | Self-declared semantic tags |
| `description` | Free text | Author-provided project summary |
| `has_wiki` / `has_pages` | Boolean | Documentation effort |
| `license` | Categorical | Professional/open-source practice |
| `archived` | Boolean | Active vs. abandoned project |

---

## How were repository summaries created?

A deterministic **text representation** was built for each repository by `src/preprocessing.py` using a fixed template with three sections:

```
[METRICS] This repository is written in {language}. It has {stars} stars,
{forks} forks, and {issues} open issues. Size: {size} KB.
Wiki enabled: {has_wiki}. Pages enabled: {has_pages}.
License: {license}. Archived status: {archived}.
[DESCRIPTION] {description}
[TOPICS] {topics}
```

This structured format was designed to be both human-readable and token-efficient for LLM prompting and BERT fine-tuning. Null or missing values are substituted with explicit defaults (`"None"`, `"No description provided."`) to avoid empty token sequences.

---

## How were prompts designed?

Weak labeling (`src/llm_labeling.py`) uses the **DeepSeek** API (`deepseek-chat` model) with an OpenAI-compatible client:

- **System prompt:** Positions the model as an *expert Software Engineering Evaluator* and defines the three categories with concrete distinguishing criteria. Temperature is set to `0.0` for deterministic, reproducible labels.
- **User turn:** The full `text_representation` string for the repository is passed directly — no additional wrapping.
- **Output format:** The model is instructed to return a strict JSON object `{"label": "<category>"}`. Responses that do not match one of the three valid labels are retried up to 3 times before being discarded.

This zero-shot approach avoids the need for any seed examples while still producing consistent labels at scale.

---

## How was the dataset split?

A **stratified two-step split** was applied (via `src/splitting.py`) to preserve class proportions in every partition:

| Split | Size | Rows | Purpose |
|---|---|---|---|
| Train | 70 % | ~315 | Model fine-tuning |
| Validation | 15 % | ~68 | Early stopping, checkpoint selection |
| Test | 15 % | ~68 | Final unbiased evaluation |

**Class distribution after splitting:**

| Class | Train | Val | Test |
|---|---|---|---|
| Low-value / Template | ~207 | ~44 | 45 |
| Senior / Lead | ~92 | ~20 | 20 |
| Intern / Junior | ~15 | ~4 | 3 |

The severe imbalance in the Intern/Junior class (~5 % of total) was addressed during training via **per-class weights** computed with `sklearn.utils.class_weight.compute_class_weight`, injected into a custom `WeightedLossTrainer`. Gathering more data was not feasible under project time constraints.

---

## Which BERT model was used?

**`distilbert-base-uncased`** — a distilled version of BERT-base that is 40 % smaller and 60 % faster while retaining ~97 % of BERT's performance on classification tasks.

| Hyperparameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Max sequence length | 256 tokens |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Train batch size | 16 (GPU) / 8 (CPU) |
| Evaluation strategy | Per epoch |
| Best checkpoint metric | Macro F1 |
| Mixed precision (fp16) | Yes (when CUDA available) |

The fine-tuned model and tokenizer are saved to `models/trained_models/distilbert_repo_classifier/`.

---

## What were the final metrics?

Evaluated on the held-out test set (68 repositories):

### Macro averages

| Metric | Score |
|---|---|
| **Accuracy** | **0.853** |
| Macro F1 | 0.572 |
| Macro Precision | 0.552 |
| Macro Recall | 0.606 |

### Per-class breakdown

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low-value / Template | 0.95 | 0.87 | **0.91** | 45 |
| Senior / Lead / Architect | 0.70 | 0.95 | **0.81** | 20 |
| Intern / Junior-level | 0.00 | 0.00 | **0.00** | 3 |

The 0.00 F1 on Intern/Junior is a **data scarcity issue**: with only 3 test examples, a single misclassification collapses recall to 0 %. The model conflates Intern repos with Low-value ones due to shared surface features (low stars, brief descriptions). This is the primary limitation of the current dataset.

---

## What are the main limitations?

1. **Intern/Junior class scarcity.** Only 22 labeled examples (5 % of total) and 3 test examples make it impossible to evaluate or improve the model on this class without collecting more data.
2. **Weak label noise.** DeepSeek labels are imperfect — the model has no access to the actual code, commit history, or contributor profile. Repositories with ambiguous descriptions may be mislabeled.
3. **Text-only features.** The classifier sees only metadata, not the codebase itself. A well-described empty project can fool the model; a poorly described production system can be underrated.
4. **Domain drift.** The collection queries embed strong bias (star counts, keywords). The model may not generalize to repositories collected with different strategies.
5. **Static snapshot.** Repository quality changes over time; a low-activity repo today may be actively developed tomorrow.

---

## What are the possible business applications?

| Use case | Description |
|---|---|
| **Resume pre-screening** | Automatically rank candidates by the quality of their submitted GitHub portfolio before human review |
| **ATS integration** | Plug into Applicant Tracking Systems (Greenhouse, Lever) as an enrichment step on the GitHub URL field |
| **Recruiter dashboard** | Give non-technical recruiters a simple "signal score" per repository to guide conversations |
| **Sourcing & outreach** | Proactively identify Senior/Lead-tier engineers on GitHub for cold outreach |
| **Bootcamp/program assessment** | Help coding bootcamps or universities evaluate student project submissions at scale |
| **Contractor vetting** | Quickly assess freelancer portfolios before engaging for a contract |

---

## How to run the project?

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv
- A GitHub personal access token (`GITHUB_TOKEN`)
- A DeepSeek API key (`DEEPSEEK_API_KEY`)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-user>/github_hiring_repository_intelligence
cd github_hiring_repository_intelligence

# Create and activate the conda environment
conda create -n github_hiring_intel python=3.10 -y
conda activate github_hiring_intel

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env   # then edit .env with your keys
```

### Pipeline stages

Run each stage in order from the project root:

```bash
# Stage 1 – Collect repositories from the GitHub API
python src/github_collector.py

# Stage 2 – Build text representations from raw metadata
python src/preprocessing.py

# Stage 3 – Weak labeling with DeepSeek
python src/llm_labeling.py

# Stage 4 – Stratified train/val/test split
python src/splitting.py

# Stage 5 – Fine-tune DistilBERT
python src/train.py

# Stage 6 – Evaluate the fine-tuned model
python src/evaluation.py
```

**Google Colab:** Mount your Drive, `cd` to the project root, and prefix each command with `!`. See the docstring at the top of each script for the exact Colab setup block.

---

## How to run the Streamlit app?

```bash
# Make sure the conda environment is activated
conda activate github_hiring_intel

# From the project root
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

The dashboard has four tabs:

| Tab | Content |
|---|---|
| 📄 Problem & Methodology | Technical report: objective, signals, prompt strategy, dataset limitations |
| 📊 Exploratory Analysis | KPIs, interactive Plotly charts, analytical commentary |
| 🎯 Model Results | Metrics, confusion matrix, per-class breakdown, error analysis |
| 🔎 Interactive Exploration | Filter test-set repos, inspect metadata, run live DistilBERT inference |

**Google Colab:**

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/<your_path>/github_hiring_repository_intelligence

!pip install streamlit pyngrok -q
from pyngrok import ngrok
import subprocess, threading

def _run():
    subprocess.run(["streamlit", "run", "app.py",
                    "--server.port=8501", "--server.headless=true"])
threading.Thread(target=_run, daemon=True).start()
print("Public URL:", ngrok.connect(8501))
```

---

## Project Structure

```
github_hiring_repository_intelligence/
│
├── app.py                          # Streamlit dashboard
├── README.md
├── requirements.txt
│
├── src/
│   ├── github_collector.py         # Stage 1 – GitHub API collection
│   ├── preprocessing.py            # Stage 2 – Text representation builder
│   ├── summarization.py            # Stage 3a – Repository summarization
│   ├── llm_labeling.py             # Stage 3b – DeepSeek weak labeling
│   ├── splitting.py                # Stage 4 – Stratified train/val/test split
│   ├── train.py                    # Stage 5 – DistilBERT fine-tuning
│   ├── evaluation.py               # Stage 6 – Metrics & error analysis
│   ├── visualization.py            # Supporting charts
│   └── utils.py                    # Shared helpers
│
├── data/
│   ├── raw/                        # JSONL + CSV from GitHub API
│   ├── processed/                  # text_representations.csv
│   ├── labeled/                    # llm_labeled.csv (DeepSeek output)
│   └── splits/                     # train.csv / val.csv / test.csv
│
├── models/
│   └── trained_models/
│       └── distilbert_repo_classifier/   # Final model + tokenizer
│
├── output/
│   ├── figures/                    # confusion_matrix.png
│   ├── tables/                     # misclassified_examples.csv
│   └── metrics/                    # classification_report.json, evaluation_summary.txt
│
└── video/
    └── link.txt
```
