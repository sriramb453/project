**SEC 10-K Sentiment Classification Pipeline**

A modular, production-ready machine learning pipeline that preprocesses and classifies SEC 10-K textual disclosures. The repository contains a training pipeline, evaluation tools, and a FastAPI-based prediction endpoint that serves the production model.

---

## Repository Layout

- **Project root:** top-level project folder containing the runnable environment and artifacts.
- **Files & folders:**
  - `/.venv/`: local virtual environment (created by setup scripts).
  - [api/app.py](api/app.py): FastAPI application exposing `/predict`.
  - `/data/`: raw inputs and sample payloads used during development.
  - `/models/`: serialized artifacts loaded by the API (`best_model.joblib`, `tfidf_vectorizer.joblib`).
  - `/src/`: training and preprocessing code (`train.py`, `preprocess.py`, `features.py`, `evaluate.py`, `utils.py`).
  - `requirements.txt`: pinned Python dependencies for reproducible setups.

---

## Quick Start

Prerequisites: Python 3.9+ and git.

Create a venv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place trained artifacts (if available):

```bash
mkdir -p models
cp data/best_boosting_model.joblib models/best_model.joblib
cp data/tfidf_vectorizer.joblib models/tfidf_vectorizer.joblib
```

Run the API (development):

```bash
./.venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
```

---

## API Usage

- **Prediction endpoint:** POST `/predict`
  - Request JSON: `{ "text": "<document text>" }`
  - Response JSON: `{ "label": <int>, "confidence": <float> }`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Sample document text"}'
```

---

## Model Benchmarks

The pipeline evaluates three gradient-boosting models on a held-out test split (~20% of n=3000):

| Model | Global Accuracy | F1-Score (Weighted) | Operational Profile |
|---|---:|---:|---|
| AdaBoost | 64.10% | 0.6455 | Higher variance; sensitive to textual anomalies. |
| XGBoost | 82.28% | 0.8186 | Best overall performance; selected for production. |
| CatBoost | 80.89% | 0.8034 | Robust to overfitting; resilient to regularized trees. |

---

## Concise Report — Preprocessing, Features, Model Comparison, Selection

### Summary
This section documents preprocessing choices, feature engineering, model comparison, and the rationale for selecting the production model used by the project.

### 1. Preprocessing & Text Cleaning
Source: implementation in `src/preprocess.py` (`clean_sec_text`, `get_sentiment`).

- Input validation: non-string inputs return an empty string.
- Normalization: lowercase conversion to reduce vocabulary sparsity.
- HTML removal: strip tags via regex `r'<[^>]+>'` to remove markup.
- Character filtering: remove non-alphabetic characters with `r'[^a-zA-Z\s]'` (keeps letters and whitespace only).
- Whitespace normalization: collapse repeated whitespace to a single space and trim.
- Pseudo-labeling: compute sentiment polarity with `TextBlob` and map polarity to 3 classes using thresholds (polarity < -0.05 → negative (0); > 0.05 → positive (2); otherwise neutral (1)).
- Filtering: drop rows with empty cleaned text prior to split.
- Train/test split: stratified split (80/20) with fixed `random_state=42` for reproducibility.

Rationale: these steps target noise common in SEC filings (HTML remnants, punctuation, numeric tokens) and produce compact, normalized text for robust vectorization. Using `TextBlob` polarity for pseudo-labels provides a lightweight, language-aware seeding mechanism for weak supervision.

### 2. Features Used
Source: `src/features.py` (`build_tfidf_features`).

- Main representation: TF–IDF (`TfidfVectorizer`) with English stop words removed (`stop_words='english'`).
- Dimensionality control: `max_features=1000` (default), limiting vocabulary to the most informative tokens.
- Output: dense arrays produced via `.toarray()` to feed the chosen boosting classifiers.

Rationale: TF–IDF captures term importance and downweights common tokens — an effective, interpretable baseline for text classification with gradient-boosted trees. Limited vocabulary keeps model size, training time, and overfitting risk manageable on modest datasets (~3k samples).

### 3. Model Comparison — Metrics & Interpretation
Source: training orchestration in `src/train.py` and evaluation in `src/evaluate.py` (metrics: Accuracy, F1-weighted).

The training pipeline evaluated three boosting-family classifiers on an out-of-sample test split (n ≈ 600):

| Model | Global Accuracy | F1-Score (Weighted) |
|---|---:|---:|
| AdaBoost | 64.10% | 0.6455 |
| XGBoost | 82.28% | 0.8186 |
| CatBoost | 80.89% | 0.8034 |

Interpretation:
- XGBoost outperforms the others across both accuracy and F1-weighted, showing better class discrimination and balanced per-class performance (weighted F1 accounts for class sizes).
- CatBoost is close behind; its regularized tree boosting tends to be robust to overfitting and categorical quirks but here is slightly behind XGBoost, likely due to TF–IDF numeric inputs being well-suited to XGBoost's optimizations.
- AdaBoost trails significantly — it is more sensitive to noisy or weak features and less well-suited to multi-class TF–IDF inputs in this pipeline.

### 4. Best Model Selection & Justification
- Selected model: **XGBoost** (production artifact saved as `models/best_model.joblib`).

Justification:
- Highest weighted F1 (0.8186) — prioritizes balanced performance across classes and is robust to class imbalance.
- Highest global accuracy (82.28%) — indicates strongest overall predictive correctness on the held-out set.
- Practical considerations: XGBoost trains fast, supports parallelism (`n_jobs=-1` in training), and provides mature model export/import semantics compatible with the inference stack.

### Notes on Reproducibility & Next Steps
- To reproduce the training and metrics: run `python src/train.py` inside the activated venv. Artifacts are written by `src/utils.py` into `../models`.
- Consider improvements:
  - Replace TextBlob pseudo-labels with a human-labeled seed set or a small fine-tuned transformer to improve label quality.
  - Expand TF–IDF (n-grams) or add document-level features (length, entity counts) from `src/features.py` variants.
  - Add cross-validation and calibration (probability calibration) when deploying to production.

### Limitations
- Pseudo-labeling via polarity is imperfect for financial language and may mislabel domain-specific neutral or mixed-tone passages.
- TF–IDF + boosting is a solid baseline but may miss long-range dependencies or context that transformer models capture.

---

Generated from code in `src/` on project state as of this update.
# project
