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