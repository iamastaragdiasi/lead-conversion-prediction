# Lead Conversion Prediction Engine

A reproducible machine learning system for estimating the probability that a sales lead will convert into a customer.

The project covers the complete workflow from synthetic data generation and exploratory analysis to model training, evaluation, threshold analysis, model serialization, and FastAPI-based prediction.

---

## Project Overview

The Lead Conversion Prediction Engine predicts the probability of lead conversion using information available before the conversion outcome is known.

The system provides:

- Conversion probability
- Conversion percentage
- Binary conversion prediction
- Deployment threshold
- Conversion potential category

The final system uses a **Tuned Random Forest** model with a deployment threshold of **0.40**.

---

## Final Model

| Component | Result |
|---|---|
| Model | Tuned Random Forest |
| Raw input features | 14 |
| Processed features | 33 |
| Deployment threshold | 0.40 |
| Analysis best F1 | 0.6513 |
| Test Accuracy | 0.6483 |
| Test Precision | 0.5868 |
| Test Recall | 0.6475 |
| Test F1-score | 0.6157 |
| Test ROC-AUC | 0.7220 |

---

## Input Features

### Categorical Features

- `lead_source`
- `industry`
- `location`
- `company_size`

### Numerical Features

- `lead_age_days`
- `interactions`
- `followups`
- `response_time_hours`
- `quotation_sent`
- `quotation_value`
- `website_visits`
- `previous_customer`
- `demo_attended`
- `salesperson_experience`

`lead_id` is excluded from predictive modelling because it is an identifier rather than a predictive feature.

---

## Machine Learning Workflow

```text
Synthetic Dataset
       ↓
Data Quality Validation
       ↓
Exploratory Data Analysis
       ↓
Leakage Analysis
       ↓
Train/Test Split
       ↓
Leakage-Safe Preprocessing
       ↓
Model Training
       ↓
Model Comparison
       ↓
Random Forest Tuning
       ↓
Threshold Analysis
       ↓
Final Model
       ↓
Joblib Serialization
       ↓
Prediction Engine
       ↓
FastAPI
```

---

## Models Evaluated

The project evaluates the required classification models:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Random Forest was subsequently tuned and used as the final model.

## Final Model Selection

The tuned Random Forest was selected as the final model after comparing Logistic Regression, Decision Tree and Random Forest using the required evaluation metrics.

The selection considered predictive performance on the held-out test set, the balance between precision and recall reflected by F1-score, ROC-AUC, generalisation to unseen data, the ability to capture nonlinear relationships, feature-importance analysis, and the usefulness of conversion probabilities for lead prioritisation.

Accuracy was not treated as the sole selection criterion because it can be misleading when the target classes are imbalanced.

The final model and preprocessing artifacts are saved using Joblib and can be loaded by the prediction engine without retraining.

---

## Preprocessing

The preprocessing pipeline uses separate transformations for numerical and categorical features.

### Numerical Features

- Median imputation
- Standard scaling

### Categorical Features

- Most-frequent imputation
- One-hot encoding

---

## Threshold Analysis

The model produces a probability of conversion.

Threshold analysis identified:

- Best analysis threshold: `0.40`
- Analysis best F1-score: approximately `0.6513`

The deployment threshold is therefore `0.40`.

---

## Conversion Potential Categories

| Probability | Category |
|---|---|
| `< 0.40` | Low Conversion Potential |
| `0.40 – < 0.70` | Medium Conversion Potential |
| `>= 0.70` | High Conversion Potential |

---

## How to Train

The project includes scripts for synthetic dataset generation and data-quality validation.

From the project root:

```bash
python src/generate_data.py
python src/validate_data.py
```

---

## How to Evaluate

The complete evaluation is documented in:

`reports/evaluation_report.md`

The evaluation covers:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Model comparison
- Threshold analysis
- Feature analysis
- Final model selection
- Limitations

### Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 0.6483 |
| Precision | 0.5868 |
| Recall | 0.6475 |
| F1-score | 0.6157 |
| ROC-AUC | 0.7220 |

The deployment threshold is `0.40`, with an analysis best F1-score of approximately `0.6513`.

---

## Prediction Engine

The reusable prediction engine is implemented in:

`src/predictor.py`

It:

1. Accepts raw lead information.
2. Validates the expected features.
3. Applies the saved preprocessing pipeline.
4. Generates a conversion probability.
5. Applies the deployment threshold.
6. Produces the predicted class.
7. Assigns a conversion-potential category.

The prediction engine loads the saved artifacts and does not retrain the model during prediction.

---

## FastAPI API

A FastAPI layer exposes the prediction engine through HTTP.

Implementation:

`src/api.py`

### Endpoints

#### `GET /`

Provides basic API information.

#### `GET /health`

Checks model and prediction-service availability.

#### `POST /predict`

Accepts lead information and returns the prediction.

The API uses Pydantic validation for incoming requests.

---

## Running the API

From the project root:

```bash
uvicorn src.api:app --reload
```

---

## Model Artifacts

The final model artifacts are stored under:

`models/`

### `preprocessor.joblib`

Saved preprocessing pipeline.

### `tuned_random_forest.joblib`

Saved final Tuned Random Forest model.

### `model_metadata.json`

Contains:

- Model type
- Artifact names
- Processed feature count
- Analysis threshold
- Analysis best F1-score
- Final test metrics

---

## Project Structure

```text
lead-conversion-prediction/
│
├── data/
│   ├── raw/
│   │   └── leads_raw.csv
│   └── processed/
│       └── leads_validated.csv
│
├── models/
│   ├── preprocessor.joblib
│   ├── tuned_random_forest.joblib
│   └── model_metadata.json
│
├── notebooks/
│   └── lead_analysis.ipynb
│
├── reports/
│   ├── evaluation_report.md
│   ├── project_approach.md
│   └── figures/
│
├── src/
│   ├── api.py
│   ├── generate_data.py
│   ├── predictor.py
│   └── validate_data.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Dataset

The project uses a locally generated synthetic dataset containing more than 5,000 lead records.

The generator models realistic relationships between:

- Lead source
- Industry
- Company size
- Engagement
- Response time
- Quotation activity
- Previous-customer status
- Demo attendance
- Salesperson experience
- Conversion

The dataset intentionally contains a small amount of missing data, duplicate records, and invalid values for data-quality analysis.

No proprietary or confidential company data is required.

---

## Results

The final Tuned Random Forest achieved the following results on the held-out test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 0.6483 |
| Precision | 0.5868 |
| Recall | 0.6475 |
| F1-score | 0.6157 |
| ROC-AUC | 0.7220 |

The deployment threshold is `0.40`, with an analysis best F1-score of approximately `0.6513`.

---

## Reproducibility

The project uses a defined dependency set in:

`requirements.txt`

The trained model and preprocessing pipeline are serialized using Joblib.

The saved artifacts allow predictions to be made without retraining the model.

---

## Evaluation Report

The detailed evaluation report is available at:

`reports/evaluation_report.md`

The report documents:

- Project overview
- Dataset
- Data quality
- EDA
- Leakage analysis
- Preprocessing
- Models evaluated
- Final model
- Threshold analysis
- Test performance
- Model artifacts
- Prediction engine
- FastAPI
- API validation
- Reproducibility
- Limitations
- Conclusion

---

## Limitations

### Synthetic Dataset

The dataset is synthetic rather than real production lead data.

Therefore, the reported metrics should not be interpreted as production-level performance on real company data.

### Generalization

Real-world lead behaviour may differ from the relationships represented in the synthetic dataset.

### Model Performance

The model should be considered a predictive prototype rather than a guaranteed conversion decision system.

### Probability Interpretation

The predicted probability is a model output and should not automatically be interpreted as a perfectly calibrated real-world probability.

### Threshold

The `0.40` threshold was selected through analysis on the project data. Production deployment would require validation using representative real-world data and an appropriate business objective.

---

## Future Improvements

Possible future improvements include:

- Evaluation using representative real-world lead data
- Probability calibration
- Additional cross-validation and robustness analysis
- Further threshold validation using business objectives
- Additional justified hyperparameter experimentation
- Model monitoring after deployment
- Data-drift monitoring
- Automated model retraining
- Production deployment and scaling
- Authentication and authorization for the prediction API

These are future directions and are not required components of the current project implementation.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- SciPy
- Jupyter
- FastAPI
- Uvicorn
- Pydantic
- Git

---

## Project Status

**Completed**

The repository contains:

- Synthetic dataset generation
- Data-quality validation
- EDA and experimentation notebook
- Leakage analysis
- Leakage-safe preprocessing
- Required model experiments
- Tuned Random Forest
- Threshold analysis
- Saved model artifacts
- Prediction engine
- FastAPI prediction API
- API validation
- Evaluation report
- Complete project documentation