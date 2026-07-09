# TelcoX Retention Copilot

A churn-prediction ML pipeline paired with a 3-condition Explainable AI (XAI) research experiment on the IBM Telco Customer Churn dataset.

## Project Overview

This project has two parts that build on each other:

- **An ML pipeline** (`02-data-processing.py` → `03-models-train.py` → `04-data-selection-5.py`) that cleans the raw customer data, engineers features, trains and tunes four candidate churn-prediction models, and selects 15 representative customers to use in a study.
- **Three Streamlit dashboards** (`blackbox.py`, `standard.py`, `actionable.py`) that present those customers to study participants under three levels of AI explainability — no explanation, a SHAP explanation, or a SHAP explanation plus AI-recommended retention actions — and log every decision participants make.

The dashboards are branded **"TelcoX"** (a fictional telecom) and are designed to be reached from, and return to, an external Tivian/Unipark survey link.

> This is a research codebase: there's no packaging, automated tests, or CI. It's meant to be run as a sequence of scripts, not installed as a library.

## Project Structure

```
.
├── 01-data-analysis.ipynb            # Optional exploratory notebook (standalone)
├── 02-data-processing.py             # Step 1: clean raw data + engineer features
├── 03-models-train.py                # Step 2: train, tune, and select the best model
├── 04-data-selection-5.py            # Step 3: select 15 customers + compute SHAP/actions
├── blackbox.py                       # Streamlit app — Condition A (no explanation)
├── standard.py                       # Streamlit app — Condition B (SHAP explanation)
├── actionable.py                     # Streamlit app — Condition C (SHAP + AI actions)
├── unipark_bridge.py                 # Shared login/return bridge to the Unipark survey
├── requirements.txt
```

The scripts also expect four folders that are **not included in the repo and are not created automatically** — you'll create these during installation:

| Folder | Holds |
|---|---|
| `data/` | The raw input file(s), plus every file the pipeline produces |
| `models/` | The trained model and its supporting artifacts |
| `evaluation/` | The diagnostic report images (`.png`) |
| `results/` | Each participant's logged decisions |

## Installation

```bash
# 1. Clone and enter the repository
git clone <this-repo-url>
cd <repo-directory>

# 2. Create and activate a virtual environment (Python 3.9–3.11 recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the folders the scripts expect
mkdir -p data models evaluation results

# 5. Add the raw data
cp CustomerChurn.xlsx data/     # IBM Telco Customer Data - Raw data used by the pipeline
```

## Step-by-Step: Running the Project

Run these in order the first time — each script reads files produced by the one before it.

### Step 0 (optional) — Explore the raw data
```bash
jupyter notebook 01-data-analysis.ipynb
```
Standalone exploratory analysis on the raw CSV. Not required for the steps below, and nothing here feeds into them.

### Step 1 — Clean the data and engineer features
```bash
python 02-data-processing.py
```
- **Needs:** `data/CustomerChurn.xlsx`
- **Produces:** `data/telco-processed.parquet`, `models/engineering-stats.joblib`, `evaluation/processing-report.png`

### Step 2 — Train and select the best model
```bash
python 03-models-train.py
```
- **Needs:** `data/telco-processed.parquet` (from Step 1)
- **Produces:** `models/best-model.joblib`, `models/model-columns.joblib`, `evaluation/training-report.png`

### Step 3 — Select the 15 study customers
```bash
python 04-data-selection-5.py --strategy B
```
- **Needs:** the parquet and model files from Steps 1–2
- **Produces:** `data/cond-A-blackbox-B.parquet`, `data/cond-B-standard-xai-B.parquet`, `data/cond-C-actionable-xai-B.parquet`, `evaluation/XAI-selection-report-5-B.png`

### Step 4 — Launch a dashboard
```bash
streamlit run blackbox.py     # Condition A — no explanation
streamlit run standard.py     # Condition B — SHAP explanation
streamlit run actionable.py   # Condition C — SHAP + AI-recommended actions
```
- **Needs:** the matching `data/cond-*-B.parquet` (Step 3) and the `models/` files (Step 2)
- Visiting the URL with no `?pid=`/`?tic=` parameters shows a manual login form, for local testing without a Unipark session
- **Produces:** `results/{participant_id}-{A,B,C}.csv` once a participant completes and confirms their session
