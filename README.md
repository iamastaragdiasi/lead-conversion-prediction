# Lead Conversion Prediction Engine

**Author:** Astarag Diasi — AI/ML Intern, Codly India Pvt. Ltd.  
**Task:** AI/ML Intern Project Task 01 — Lead Conversion Prediction Engine  
**Submitted:** 1 October 2026\
**AI assistance disclosure:** Claude (Anthropic) was used as a coding assistant during development; some commits are attributed to it. All design decisions, methodology and results were reviewed and are understood by the author. No AI/LLM API is used inside the project itself.

A reproducible machine learning pipeline that estimates the probability that a sales lead will convert into a customer, and turns it into a prediction and a Low / Medium / High potential category.

```text
Conversion Probability : 69.4%
Prediction             : Likely to Convert
Category               : MEDIUM POTENTIAL
```

The prediction comes from a trained model, not from hand-written rules. No external LLM APIs, paid AI services, API keys or company data are used.

---

## Project Overview

A business receives many leads and has limited sales time. This project learns from historical (synthetic) lead data which leads are most likely to convert, so the team can call the best leads first.

| Item | Value |
|---|---|
| Final model | **Logistic Regression** (preprocessing + model saved as one Joblib pipeline) |
| Model selection | Highest 5-fold cross-validated ROC-AUC on the training set |
| Decision threshold | **0.32** (maximum F1 on out-of-fold training predictions; test set not used) |
| Raw input features | 14 (33 after encoding) |
| Dataset | 6,000 cleaned synthetic leads (6,020 raw), 43.8% converted |
| Test ROC-AUC | 0.7492 |
| Test recall / precision / F1 at 0.32 | 0.8267 / 0.5543 / 0.6636 |

Workflow:

```text
Generate data → Validate & clean → EDA → Leakage review → Stratified 80/20 split
→ Pipeline (impute + scale + one-hot + model) → 5-fold CV model comparison
→ Select model → Select threshold (training data only) → Final test evaluation
→ Feature importance → Save pipeline (Joblib) → CLI predictor / FastAPI
```

---

## Requirements

- Python 3.11+ (developed with Python 3.13; also tested on 3.11)
- Libraries in `requirements.txt`: pandas, NumPy, scikit-learn, Joblib, SciPy, Matplotlib, Seaborn, Jupyter
  - scikit-learn is pinned to **1.9.1** because the saved model file was trained with it; other versions cannot load it reliably. scikit-learn 1.9 needs Python 3.11 or newer.
  - The other libraries use minimum versions.
- Optional, for the web API only: FastAPI, Uvicorn, Pydantic in `requirements-api.txt`

---

## Installation

```bash
git clone https://github.com/iamastaragdiasi/lead-conversion-prediction.git
cd lead-conversion-prediction

python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Only if you want the optional REST API:

```bash
pip install -r requirements-api.txt
```

All commands below are run from the project root.

---

## Dataset

The dataset is **synthetic** and generated locally by `src/generate_data.py` (seed 42), so anyone can recreate it exactly. The generated files are also committed in `data/`.

- **Size:** 6,020 raw records → 6,000 after removing 20 exact duplicates.
- **Target:** `converted` (1 = converted, 0 = not converted), 43.8% positive.
- **How conversion is generated:** a logistic function of lead source, industry, company size, interactions, follow-ups, response time, quotation sent and value, website visits, previous-customer status, demo attendance and salesperson experience, plus random noise. It is not random numbers.
- **Funnel relationships between features:** the features are generated as a chain, as in a real sales process — larger companies and website leads have more interactions, interactions drive follow-ups and demo attendance, a demo makes a quotation much more likely (71% vs 29%), bigger companies receive bigger quotations, and experienced salespeople respond faster.
- **Injected data-quality issues** (to demonstrate cleaning): about 1% missing values in five columns, 10 negative lead ages, 10 negative response times and 20 duplicate rows.

| Categorical features | Numerical features |
|---|---|
| `lead_source`, `industry`, `location`, `company_size` | `lead_age_days`, `interactions`, `followups`, `response_time_hours`, `quotation_sent`, `quotation_value`, `website_visits`, `previous_customer`, `demo_attended`, `salesperson_experience` |

`company_size` is stored as a band (Small / Medium / Large / Enterprise) rather than an exact head-count, because CRMs usually capture company size as a range and the band is easier for sales staff to enter reliably.

`lead_id` is an identifier and is excluded from modelling.

**Prediction point:** the model scores a lead **mid-funnel**, after the first sales activity, using feature values as of the scoring date (interactions, follow-ups, demo and quotation status so far). It is not designed for brand-new leads with no activity yet. See section 5 of the evaluation report.

---

## How to Train

```bash
python src/generate_data.py    # creates data/raw/leads_raw.csv (6,020 rows)
python src/validate_data.py    # creates data/processed/leads_validated.csv (6,000 rows)
python src/train.py            # trains, compares and saves the model (about 1–2 minutes)
```

`src/train.py`:

1. Splits the data 80/20, stratified on `converted`.
2. Builds one scikit-learn `Pipeline` per model (median imputation + scaling for numbers, most-frequent imputation + one-hot encoding for categories, then the classifier), so preprocessing is fitted on training data only.
3. Compares **Logistic Regression, Decision Tree, Random Forest** and a **GridSearchCV-tuned Random Forest** with 5-fold cross-validation on the training set.
4. Selects the model with the highest cross-validated ROC-AUC.
5. Selects the decision threshold from out-of-fold training predictions (maximum F1).
6. Evaluates every model once on the untouched test set.
7. Saves:
   - `models/lead_conversion_pipeline.joblib` (preprocessing + model)
   - `models/model_metadata.json` (features, threshold, categories, metrics)
   - `reports/metrics.json` (all model results and feature importance)
   - figures in `reports/figures/`

The notebook `notebooks/lead_analysis.ipynb` contains the full EDA and runs the same modelling code step by step (**Kernel → Restart & Run All**).

---

## How to Evaluate

Evaluate the saved model on the held-out test set **without retraining**:

```bash
python src/evaluate.py
```

It loads `models/lead_conversion_pipeline.joblib`, rebuilds the same stratified test split, and prints accuracy, precision, recall, F1, ROC-AUC, Brier score and the confusion matrix at the decision threshold (0.32) and at 0.50, plus the actual conversion rate of each potential category. It also checks the results against the metrics saved at training time.

Other evaluation outputs:

- Training prints the comparison table and final metrics to the console.
- `reports/evaluation_report.md` is the full evaluation: metric definitions, why accuracy is not enough, leakage review, model comparison, confusion matrices, model selection, threshold analysis, feature importance and limitations.
- `reports/metrics.json` holds all the numbers.
- `reports/figures/` holds the ROC curves, precision-recall curves, confusion matrices, threshold analysis, permutation importance and coefficient plots.

---

## How to Predict

The predictor loads the saved pipeline. It never retrains.

```bash
python src/predict.py              # interactive: asks for each lead detail
python src/predict.py --example    # scores a built-in example lead
python src/predict.py --input lead.json
python src/predict.py --csv leads.csv   # batch scoring → leads_scored.csv
```

Example `lead.json`:

```json
{
  "lead_source": "Website", "industry": "Retail", "location": "Bengaluru",
  "company_size": "Medium", "lead_age_days": 20, "interactions": 6, "followups": 3,
  "response_time_hours": 2.0, "quotation_sent": 1, "quotation_value": 85000,
  "website_visits": 5, "previous_customer": 0, "demo_attended": 1,
  "salesperson_experience": 6
}
```

Example output:

```text
============================================
       LEAD CONVERSION PREDICTOR
============================================
Lead source             : Website
Industry                : Retail
Location                : Bengaluru
Company size            : Medium
Lead age                : 20
Interactions            : 6
Follow-ups              : 3
First response time     : 2.0
Quotation sent          : Yes
Quotation value         : 85000
Website visits          : 5
Previous customer       : No
Demo attended           : Yes
Salesperson experience  : 6
--------------------------------------------
Conversion Probability  : 69.4%
Prediction              : Likely to Convert
Category                : MEDIUM POTENTIAL
Decision threshold      : 0.32
Model                   : Logistic Regression
============================================
```

From Python:

```python
from src.predictor import predict_lead
predict_lead({...})   # returns probability, percentage, class, label, threshold, category
```

### Optional: REST API

```bash
pip install -r requirements-api.txt   # once
uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000/docs for the Swagger UI. Endpoints: `GET /`, `GET /health`, `POST /predict`. Invalid input returns HTTP 422.

### Conversion categories

| Probability | Category | Basis |
|---|---|---|
| < 0.32 | Low Potential | Below the analysed decision threshold |
| 0.32 – < 0.70 | Medium Potential | Between the threshold and 0.70 |
| ≥ 0.70 | High Potential | Business convention, **not** statistically derived |

On the test set, High / Medium / Low leads actually converted at 83% / 47% / 22%.

---

## Project Structure

```text
lead-conversion-prediction/
├── data/
│   ├── raw/leads_raw.csv                  # generated raw data (with quality issues)
│   └── processed/leads_validated.csv      # cleaned data used for modelling
├── models/
│   ├── lead_conversion_pipeline.joblib    # preprocessing + final model
│   └── model_metadata.json                # threshold, features, metrics
├── notebooks/
│   └── lead_analysis.ipynb                # EDA + modelling experiments
├── reports/
│   ├── evaluation_report.md               # full evaluation report
│   ├── project_approach.md                # approach document
│   ├── metrics.json                       # all model metrics
│   └── figures/                           # EDA and evaluation charts
├── src/
│   ├── generate_data.py                   # synthetic data generator
│   ├── validate_data.py                   # data-quality checks and cleaning
│   ├── train.py                           # training, comparison, selection, saving
│   ├── evaluate.py                        # evaluates the saved model on the test set
│   ├── predictor.py                       # prediction engine (loads saved pipeline)
│   ├── predict.py                         # command-line predictor
│   └── api.py                             # optional FastAPI service
├── requirements.txt                       # core dependencies
├── requirements-api.txt                   # optional API dependencies
└── README.md
```

---

## Results

Test set (1,200 leads). Model comparison at the default 0.50 threshold; CV = 5-fold on the training set:

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.7415** | 0.6925 | 0.6765 | 0.5695 | 0.6184 | **0.7492** |
| Decision Tree | 0.6715 | 0.6608 | 0.6261 | 0.5581 | 0.5901 | 0.6977 |
| Random Forest | 0.7267 | 0.6783 | 0.6257 | 0.6590 | 0.6419 | 0.7434 |
| Tuned Random Forest | 0.7268 | 0.6717 | 0.6193 | 0.6476 | 0.6331 | 0.7430 |

Final model (Logistic Regression) at the decision threshold of **0.32**:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.6333 | 0.5543 | 0.8267 | 0.6636 | 0.7492 |

It finds 434 of 525 real buyers (83%). The cost is lower precision: more calls to leads that do not buy. That trade-off is intentional, because a missed deal costs more than an extra call.

Why Logistic Regression: highest cross-validated and test ROC-AUC, almost no overfitting (the Random Forest overfits: train 0.83 vs CV 0.73), probabilities whose calibration is measured in `reports/calibration_analysis.md`, fully interpretable and very cheap to run. Tuning the Random Forest did not improve it. At their own thresholds the Random Forests reach a slightly higher test F1 (0.669 vs 0.664), a difference too small to outweigh these advantages.

Most important features (permutation importance): demo attended, quotation sent, response time, lead source, previous customer. Location and lead age contribute nothing; interactions and website visits contribute almost nothing on their own because their effect runs through follow-ups, demos and quotations.

---

## Limitations

- **Synthetic data:** the results reflect the generator's assumptions, not a real market, and are not production performance.
- **Modest accuracy:** ROC-AUC of about 0.75 gives useful ranking, but many individual predictions will be wrong.
- **Near-linear data:** the generator is logistic, which favours Logistic Regression; real data may behave differently.
- **Threshold objective:** 0.32 maximises F1; a real business should set it from the actual cost of a missed deal versus a wasted call.
- **Quotation value imputation:** 21 leads with a sent quotation have a missing value that is imputed as 0.
- **Leakage in real data:** activity features must be captured at prediction time, not after the deal closes.
- **Prediction point:** leads are scored mid-funnel; brand-new leads with no sales activity are out of scope.
- **Probabilities** are not explicitly calibrated and will drift if customer behaviour changes.

---

## Future Improvements

- Real, time-stamped CRM data with a time-based validation split.
- Threshold chosen from business costs and sales capacity.
- Probability calibration and calibration plots.
- Gradient boosting comparison on real data.
- Model versioning, monitoring, data-drift detection and scheduled retraining.
- CRM integration for batch scoring, and API authentication.

---

## Technology Stack

Python · pandas · NumPy · scikit-learn · Joblib · Matplotlib · Seaborn · Jupyter · Git · FastAPI / Uvicorn / Pydantic (optional API)

---

## Restrictions Followed

- No OpenAI, Gemini, Claude or other external LLM APIs.
- No paid AI APIs or company-provided API keys.
- No proprietary or confidential Codly data; all data is synthetic and generated locally.

<!-- review-addendum -->
## Additional Analysis and Verification

| What | Command | Output |
|---|---|---|
| Leakage experiment and feature-availability review | `python src/leakage_demo.py` | `reports/leakage_demo.md` |
| Calibration and cost-based threshold table | `python src/calibration_analysis.py` | `reports/calibration_analysis.md` |
| Automated tests | `pip install -r requirements-dev.txt` then `pytest -q` | 8 tests |
| Model integrity check | `python src/hash_model.py --check` | hash match |
| Exact reproducible environment | `pip install -r requirements.lock.txt` | pinned versions |

See `reports/review_addendum.md` for these items:
- Why the conversion rate is 43.8%.
- Deviations from the approach document.
- The known `quotation_value` imputation issue.
- Model-file security.

**Security note:** `.joblib` files execute code when loaded. Only load model files from a trusted source.

## AI Assistance Disclosure

Claude (Anthropic), used through its chat interface, assisted with code review, the leakage and calibration analysis scripts, tests and documentation wording. Some earlier commits list Claude as author for that reason. The project itself calls no AI or LLM API. I reviewed, ran and verified all code and results, and I take responsibility for the submission.
