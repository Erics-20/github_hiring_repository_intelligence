"""
Streamlit Dashboard – GitHub Hiring Repository Intelligence
============================================================
Run locally
-----------
    streamlit run app.py

Run in Google Colab
-------------------
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/<your_path>/github_hiring_repository_intelligence

    !pip install streamlit pyngrok transformers torch -q
    from pyngrok import ngrok
    import subprocess, threading

    def _run():
        subprocess.run(["streamlit", "run", "app.py",
                        "--server.port=8501", "--server.headless=true"])
    threading.Thread(target=_run, daemon=True).start()
    print("Public URL:", ngrok.connect(8501))
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
LABELED_PATH        = Path("data/labeled/llm_labeled.csv")
TEST_PATH           = Path("data/splits/test.csv")
MODEL_DIR           = Path("models/trained_models/distilbert_repo_classifier")
REPORT_PATH         = Path("output/metrics/classification_report.json")
CONFUSION_PATH      = Path("output/figures/confusion_matrix.png")
MISCLASSIFIED_PATH  = Path("output/tables/misclassified_examples.csv")

MAX_LENGTH = 256

LABEL_COLORS = {
    "Low-value / Template":                  "#EF553B",
    "Intern / Junior-level":                 "#636EFA",
    "Senior-level / Lead / Architect-level": "#00CC96",
}
LABEL_ICONS = {
    "Low-value / Template":                  "🔴",
    "Intern / Junior-level":                 "🔵",
    "Senior-level / Lead / Architect-level": "🟢",
}


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Hiring Intelligence",
    page_icon="🔍",
    layout="wide",
)


# ── Cached Data & Model Loaders ────────────────────────────────────────────────
@st.cache_data
def load_labeled() -> pd.DataFrame:
    return pd.read_csv(LABELED_PATH)


@st.cache_data
def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


@st.cache_data
def load_report() -> dict:
    with open(REPORT_PATH) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading DistilBERT model …")
def load_model():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        device=-1,
        truncation=True,
        max_length=MAX_LENGTH,
    )


# ── App Header ─────────────────────────────────────────────────────────────────
st.title("🔍 GitHub Hiring Repository Intelligence")
st.caption(
    "Automated candidate repository evaluation using weak labeling (DeepSeek) "
    "and a fine-tuned DistilBERT classifier."
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📄  Problem & Methodology",
    "📊  Exploratory Analysis",
    "🎯  Model Results",
    "🔎  Interactive Exploration",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Problem & Methodology
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Problem & Methodology")

    with st.expander("🎯 Project Objective", expanded=True):
        st.markdown("""
        **Goal:** Build an automated NLP pipeline that evaluates public GitHub repositories
        submitted by job applicants and classifies them into one of three hiring-signal tiers:

        | Icon | Category | Signal |
        |---|---|---|
        | 🔴 | **Low-value / Template** | Forks, boilerplates, homework dumps — no meaningful hiring signal |
        | 🔵 | **Intern / Junior-level** | Learning exercises, small personal projects, limited complexity |
        | 🟢 | **Senior-level / Lead / Architect-level** | Production-ready, high-impact, well-documented repositories |

        This lets recruiters **prioritise manual review** on repositories that genuinely signal
        engineering ability, reducing screening time by automatically filtering noise.
        """)

    with st.expander("🔎 Repository Selection Methodology"):
        st.markdown("""
        Repositories were sourced via the **GitHub Search API** using bucket-targeted queries:

        - **Low-value bucket** — Repos with 0 stars, 0 forks, no description, and no license.
          Designed to capture abandoned clones and auto-generated scaffolding.
        - **Intern / Junior bucket** — Repos whose descriptions contain keywords such as
          *"homework"*, *"bootcamp"*, *"learning"*, or *"tutorial"*, combined with low engagement.
        - **Senior / Lead bucket** — Repos with **≥ 50 stars**, meaningful topic tags
          (*production*, *architecture*, *API*), and indicators of professional practice such as
          CI/CD configuration or license files.

        Each bucket was capped to stay within API rate limits and collection time constraints.
        """)

    with st.expander("📡 GitHub Signals Used"):
        st.markdown("""
        The following metadata was extracted from each repository and encoded into a structured
        natural-language string (`text_representation`) fed to the classifier:

        | Signal | Type | Hiring Rationale |
        |---|---|---|
        | `stargazers_count` | Numeric | Community validation proxy |
        | `forks_count` | Numeric | Reuse and credibility indicator |
        | `open_issues_count` | Numeric | Active maintenance signal |
        | `size` (KB) | Numeric | Rough codebase complexity proxy |
        | `language` | Categorical | Primary programming language |
        | `topics` | Text list | Self-declared semantic tags |
        | `description` | Free text | Author-provided project context |
        | `has_wiki` / `has_pages` | Boolean | Documentation effort |
        | `license` | Categorical | Professional / open-source practice |
        | `archived` | Boolean | Active vs. abandoned project |
        """)

    with st.expander("🤖 Prompt Strategy — Weak Labeling with DeepSeek"):
        st.markdown("""
        Manually labeling 450+ repositories is time-expensive. Instead, a **Weak Labeling**
        strategy was used with **DeepSeek** (accessed via the OpenAI-compatible API):

        1. Each repository's `text_representation` was passed as the user turn.
        2. A system prompt defined the three categories with concrete examples to anchor the
           model's decision boundary.
        3. The model was instructed to output strict JSON: `{"label": "<category>"}`.
        4. Temperature was set to `0.0` for deterministic, reproducible annotations.

        **Why weak labels?** They are *weak* because they carry LLM uncertainty and may
        misread ambiguous repos. However, at scale they are consistent enough to train a
        downstream classifier that can generalise beyond any single LLM call.
        """)

    with st.expander("📦 Dataset Construction & Limitations"):
        st.markdown("""
        **Stratified split — 70 / 15 / 15:**

        | Split | Rows | Purpose |
        |---|---|---|
        | Train | ~315 | Model fine-tuning |
        | Validation | ~68 | Early stopping, checkpoint selection |
        | Test | ~68 | Final unbiased evaluation |

        ---

        **⚠️ Severe Class Imbalance**

        | Class | Count | % of total |
        |---|---|---|
        | Low-value / Template | 296 | 65.8 % |
        | Senior-level / Lead | 132 | 29.3 % |
        | Intern / Junior-level | **22** | **4.9 %** |

        The Intern/Junior class represents only ~5 % of the full dataset, meaning the test set
        contains only **3 examples** for that class — making per-class F1 for Intern/Junior
        statistically unreliable (one misclassification collapses recall to 0 %).

        **Mitigation — Class Weights (chosen over data collection):**
        Under time constraints, collecting enough Intern/Junior repositories to balance the
        dataset was infeasible. Instead, per-class weights were computed with
        `sklearn.utils.class_weight.compute_class_weight` and injected into a custom
        `WeightedLossTrainer`, penalising the model more heavily for misclassifying rare
        classes. This is a partial fix: the fundamental problem is data scarcity, not
        model architecture.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Exploratory Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Exploratory Data Analysis")

    try:
        df = load_labeled()

        n_low    = (df["llm_label"] == "Low-value / Template").sum()
        n_intern = (df["llm_label"] == "Intern / Junior-level").sum()
        n_senior = (df["llm_label"] == "Senior-level / Lead / Architect-level").sum()

        # ── KPI Row ───────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Repositories", f"{len(df):,}")
        c2.metric("🔴 Low-value",       str(n_low),    f"{n_low/len(df)*100:.1f} %")
        c3.metric("🔵 Intern/Junior",   str(n_intern), f"{n_intern/len(df)*100:.1f} %")
        c4.metric("🟢 Senior/Lead",     str(n_senior), f"{n_senior/len(df)*100:.1f} %")
        c5.metric("Avg ⭐ Stars",        f"{df['stargazers_count'].mean():.1f}")

        st.divider()

        # ── Chart 1: Category Distribution ────────────────────────────────────
        st.subheader("1. Category Distribution")

        dist = df["llm_label"].value_counts().reset_index()
        dist.columns = ["Category", "Count"]
        dist["Percentage"] = (dist["Count"] / len(df) * 100).round(1)

        fig1 = px.bar(
            dist,
            x="Count",
            y="Category",
            orientation="h",
            text=dist["Percentage"].astype(str) + " %",
            color="Category",
            color_discrete_map=LABEL_COLORS,
            title="Label Distribution — Weak Labels (DeepSeek)",
            labels={"Count": "Repositories"},
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(
            showlegend=False,
            yaxis={"categoryorder": "total ascending"},
            height=300,
            margin=dict(l=10, r=60, t=50, b=10),
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.info(
            "**Why this chart?** The horizontal bar immediately exposes the imbalance that "
            "drives every downstream design decision. A naïve model that always predicts "
            "'Low-value' would score ~66 % accuracy — making raw accuracy a misleading metric. "
            "This is why **macro F1** was used as the primary evaluation metric, and why "
            "**class-weighted loss** was adopted during fine-tuning rather than standard cross-entropy."
        )

        st.divider()

        # ── Chart 2: Stars by Category ─────────────────────────────────────────
        st.subheader("2. Star Count Distribution by Category")

        df_plot = df.copy()
        df_plot["stargazers_count_capped"] = df_plot["stargazers_count"].clip(upper=300)

        fig2 = px.box(
            df_plot,
            x="llm_label",
            y="stargazers_count_capped",
            color="llm_label",
            color_discrete_map=LABEL_COLORS,
            points="outliers",
            title="Stars per Category (values capped at 300 for readability)",
            labels={
                "llm_label": "Category",
                "stargazers_count_capped": "Star Count",
            },
        )
        fig2.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

        st.info(
            "**Why this chart?** Stars are the strongest single proxy signal for repository "
            "quality in a hiring context. The box plot confirms a clear separation: Senior/Lead "
            "repos have a substantially higher median and a long upper tail, validating that the "
            "API's star-threshold query strategy correctly captured high-signal repositories. "
            "Low-value repos cluster heavily at 0, confirming the collection logic. "
            "This also explains why the model performs well on both extremes but struggles on "
            "Intern/Junior repos, which occupy an ambiguous mid-range."
        )

        st.divider()

        # ── Chart 3: Language Distribution ────────────────────────────────────
        st.subheader("3. Top 10 Programming Languages by Category")

        top_langs = df["language"].value_counts().head(10).index.tolist()
        lang_df = (
            df[df["language"].isin(top_langs)]
            .groupby(["language", "llm_label"])
            .size()
            .reset_index(name="Count")
        )

        fig3 = px.bar(
            lang_df,
            x="language",
            y="Count",
            color="llm_label",
            color_discrete_map=LABEL_COLORS,
            barmode="stack",
            title="Repository Count — Top 10 Languages, Stacked by Category",
            labels={"language": "Language", "llm_label": "Category"},
        )
        fig3.update_layout(height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

        st.info(
            "**Why this chart?** Language choice is a weak but non-trivial hiring signal. "
            "Python dominates across all tiers (data science, automation, web). TypeScript and "
            "JavaScript lean heavily toward Senior/Lead repos, reflecting production "
            "front-end/back-end work. C++ and Rust appear almost exclusively in the Senior tier, "
            "consistent with the higher technical ceiling of systems programming. "
            "This validates the decision to include `language` as a feature in `text_representation`."
        )

    except FileNotFoundError:
        st.warning(
            f"`{LABELED_PATH}` not found. "
            "Ensure the labeling stage has been run before launching this dashboard."
        )
    except Exception as e:
        st.warning(f"Could not load exploratory data: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Model Results
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Results & Error Analysis")

    # ── Macro Metrics ─────────────────────────────────────────────────────────
    try:
        report = load_report()
        macro  = report["macro avg"]
        acc    = report["accuracy"]

        st.subheader("Macro-Average Metrics (test set, n = 68)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",        f"{acc:.3f}")
        m2.metric("Macro F1",        f"{macro['f1-score']:.3f}")
        m3.metric("Macro Precision", f"{macro['precision']:.3f}")
        m4.metric("Macro Recall",    f"{macro['recall']:.3f}")

        st.divider()

        # ── Per-Class Table ────────────────────────────────────────────────────
        st.subheader("Per-Class Performance")

        class_order = [
            "Low-value / Template",
            "Intern / Junior-level",
            "Senior-level / Lead / Architect-level",
        ]
        rows = [
            {
                "Category":  cls,
                "Precision": round(report[cls]["precision"], 3),
                "Recall":    round(report[cls]["recall"],    3),
                "F1-score":  round(report[cls]["f1-score"],  3),
                "Support":   int(report[cls]["support"]),
            }
            for cls in class_order
        ]
        perf_df = pd.DataFrame(rows)
        st.dataframe(
            perf_df.style
                .highlight_max(subset=["F1-score"], color="#c6efce")
                .highlight_min(subset=["F1-score"], color="#ffc7ce"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("""
        **Reading the results:**

        - 🔴 **Low-value / Template — F1 = 0.91:** The model excels here. This class
          dominates training (65.8 %), giving DistilBERT ample signal to learn its patterns
          (0-star, no-description, boilerplate language).
        - 🟢 **Senior / Lead — F1 = 0.81:** Strong performance despite being a minority class.
          High-star, topic-rich repositories produce distinctive text representations that the
          model separates cleanly from Low-value noise.
        - 🔵 **Intern / Junior — F1 = 0.00:** Complete failure. With only **3 test examples**,
          a single misclassification drives recall to 0 %. This is a **data scarcity problem**:
          the model conflates Intern repos with Low-value ones because they share surface features
          (low stars, brief descriptions). Class weighting partially corrected the training
          distribution but cannot overcome an evaluation set this small.
        """)

    except FileNotFoundError:
        st.warning(
            f"`{REPORT_PATH}` not found. Run `python src/evaluation.py` to generate metrics."
        )
    except Exception as e:
        st.warning(f"Could not load classification report: {e}")

    st.divider()

    # ── Confusion Matrix ───────────────────────────────────────────────────────
    st.subheader("Confusion Matrix")
    try:
        st.image(str(CONFUSION_PATH), use_container_width=True)
        st.caption(
            "Rows = true label · Columns = predicted label. "
            "All 3 Intern/Junior examples were misclassified into adjacent categories."
        )
    except Exception:
        st.warning(
            f"Confusion matrix not found at `{CONFUSION_PATH}`. "
            "Run `python src/evaluation.py` to generate it."
        )

    st.divider()

    # ── Error Analysis ─────────────────────────────────────────────────────────
    st.subheader("Error Analysis — Misclassified Examples")
    try:
        mis = pd.read_csv(MISCLASSIFIED_PATH)
        st.markdown(
            f"**{len(mis)} of 68** test repositories were misclassified "
            f"({len(mis) / 68 * 100:.1f} % error rate)."
        )
        st.dataframe(
            mis[["full_name", "true_label", "predicted_label", "text_representation"]],
            use_container_width=True,
            height=280,
        )
        st.info(
            "**Error pattern:** Most errors are boundary cases between adjacent tiers — "
            "repositories with moderate star counts (10–40) that DeepSeek labeled as Senior but "
            "the model predicts as Low-value, or template repositories with a description that "
            "superficially resembles a Junior project. The Intern/Junior misclassifications "
            "confirm the model has not learned a reliable boundary for this class in the current "
            "data regime — the fix is collecting more labeled examples, not tuning the model."
        )
    except FileNotFoundError:
        st.warning(
            f"`{MISCLASSIFIED_PATH}` not found. Run `python src/evaluation.py` to generate it."
        )
    except Exception as e:
        st.warning(f"Could not load misclassified examples: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Interactive Repository Exploration
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Interactive Repository Exploration")
    st.markdown(
        "Filter repositories from the test set, inspect their metadata, "
        "and run live inference with the fine-tuned DistilBERT model."
    )

    try:
        test_df = load_test()

        # ── Filters ──────────────────────────────────────────────────────────
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            cat_filter = st.selectbox(
                "Filter by True Category",
                options=["All"] + sorted(test_df["llm_label"].unique().tolist()),
            )
        with fc2:
            search = st.text_input(
                "Search by repository name",
                placeholder="e.g.  django   rust   portfolio …",
            )

        filtered = test_df.copy()
        if cat_filter != "All":
            filtered = filtered[filtered["llm_label"] == cat_filter]
        if search.strip():
            filtered = filtered[
                filtered["full_name"].str.contains(search.strip(), case=False, na=False)
            ]

        if filtered.empty:
            st.warning("No repositories match the current filters.")
            st.stop()

        selected_name = st.selectbox(
            f"Select a repository  ({len(filtered)} matches)",
            options=filtered["full_name"].tolist(),
        )

        row = filtered[filtered["full_name"] == selected_name].iloc[0]

        st.divider()

        left, right = st.columns([3, 2])

        # ── Metadata Card ─────────────────────────────────────────────────────
        with left:
            st.subheader(f"📁  {row['full_name']}")
            desc = row.get("description", "")
            st.markdown(
                f"_{desc if pd.notna(desc) and str(desc).strip() else 'No description provided.'}_"
            )
            url = row["html_url"]
            st.markdown(f"🔗 [{url}]({url})")

            km1, km2, km3, km4 = st.columns(4)
            km1.metric("⭐ Stars",   int(row["stargazers_count"]))
            km2.metric("🍴 Forks",   int(row["forks_count"]))
            km3.metric("🐛 Issues",  int(row["open_issues_count"]))
            lang = row.get("language")
            km4.metric("💻 Language", lang if pd.notna(lang) else "N/A")

            topics_raw = row.get("topics", "")
            if pd.notna(topics_raw) and str(topics_raw).strip():
                tags = str(topics_raw).split("|")
                st.markdown("**Topics:** " + "  ".join(f"`{t.strip()}`" for t in tags if t.strip()))

        # ── Prediction Panel ──────────────────────────────────────────────────
        with right:
            true_label = row["llm_label"]
            st.markdown("**True Label (DeepSeek weak label):**")
            st.markdown(
                f"#### {LABEL_ICONS.get(true_label, '⚪')}  {true_label}"
            )

            st.divider()

            st.markdown("**Live Inference:**")
            if st.button("🤖  Predict with DistilBERT", type="primary"):
                try:
                    with st.spinner("Running inference …"):
                        classifier  = load_model()
                        result      = classifier(str(row["text_representation"]))
                        pred_label  = result[0]["label"]
                        confidence  = result[0]["score"]

                    st.markdown("**Predicted Label:**")
                    st.markdown(
                        f"#### {LABEL_ICONS.get(pred_label, '⚪')}  {pred_label}"
                    )
                    st.progress(
                        float(confidence),
                        text=f"Confidence: {confidence:.1%}",
                    )

                    if pred_label == true_label:
                        st.success("✅  Prediction matches the true label.")
                    else:
                        st.error(
                            f"❌  Mismatch — model predicted **{pred_label}** "
                            f"but true label is **{true_label}**."
                        )
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")

        # ── Text Representation Preview ───────────────────────────────────────
        st.divider()
        with st.expander("📝  Full text_representation sent to the model"):
            st.code(str(row["text_representation"]), language="text")

    except FileNotFoundError:
        st.warning(
            f"`{TEST_PATH}` not found. "
            "Ensure the data splitting stage has been run."
        )
    except Exception as e:
        st.warning(f"Unexpected error loading test data: {e}")
