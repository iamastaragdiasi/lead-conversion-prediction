# Lead Conversion Prediction Engine — Evaluation Report

## 1. Project Overview

The Lead Conversion Prediction Engine is a reproducible Machine Learning system designed to estimate the probability that a sales lead will convert into a customer.

The system uses information available before the conversion outcome is known and produces:

- Conversion probability
- Predicted conversion class
- Operational conversion-potential category

The project uses locally generated synthetic data and does not depend on proprietary or confidential company data.

The implementation follows the planned workflow:

**Dataset generation → Data validation → EDA → Leakage analysis → Preprocessing → Model training → Model comparison → Model tuning → Threshold analysis → Model serialization → Prediction engine → FastAPI API**

---

## 2. Dataset

The project uses a locally generated synthetic lead dataset containing more than 5,000 records.

The dataset generator was designed to model realistic relationships between lead characteristics, engagement activity, sales activity, and conversion.

### Raw Features

The model uses 14 predictive input features:

1. `lead_source`
2. `industry`
3. `location`
4. `company_size`
5. `lead_age_days`
6. `interactions`
7. `followups`
8. `response_time_hours`
9. `quotation_sent`
10. `quotation_value`
11. `website_visits`
12. `previous_customer`
13. `demo_attended`
14. `salesperson_experience`

The `lead_id` field is treated as an identifier and is excluded from predictive modelling.

### Target

The target variable is:

`converted`

where:

- `1` = Converted
- `0` = Not Converted

The model estimates the probability of the positive class.

---

## 3. Data Quality and Preprocessing

The synthetic dataset intentionally contains a small amount of realistic data-quality problems so that the project can demonstrate proper validation.

The generated data includes:

- Missing values
- Duplicate records
- Invalid numeric values
- Realistic categorical variation

The validation process:

1. Identifies invalid numeric values.
2. Converts invalid values into missing values where appropriate.
3. Removes exact duplicate records.
4. Produces the validated dataset used for modelling.

The validated dataset is stored under:

`data/processed/leads_validated.csv`

---

## 4. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

- Dataset structure
- Missing values
- Duplicate records
- Target distribution
- Categorical feature distributions
- Numerical feature distributions
- Relationships between features and conversion
- Correlations between numerical variables
- Potential outliers

Generated analysis figures include:

- Industry distribution
- Company-size distribution
- Categorical feature relationships with conversion
- Correlation matrix
- Threshold analysis

The EDA was used to understand the dataset before model training rather than relying only on model metrics.

---

## 5. Leakage Analysis

Data leakage was considered during the project because the model must only use information that would reasonably be available when making a lead-conversion prediction.

The identifier `lead_id` was excluded because it is not a meaningful predictive feature.

The preprocessing workflow was also designed so that learned transformations are fitted using training data and subsequently applied to unseen data.

This reduces the risk of information from the test set influencing model training.

---

## 6. Preprocessing Pipeline

The final preprocessing pipeline separates numerical and categorical features.

### Numerical Features

The numerical variables use:

- Median imputation
- Standard scaling

### Categorical Features

The categorical variables use:

- Most-frequent imputation
- One-hot encoding

A scikit-learn `ColumnTransformer` is used to combine the transformations.

The preprocessing pipeline produces:

**33 processed features from 14 raw input features.**

The preprocessing object is saved as:

`models/preprocessor.joblib`

This allows the same transformations to be reused during prediction without retraining.

---

## 7. Models Evaluated

The project evaluated the required classification approaches:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Logistic Regression provides an interpretable baseline.

Decision Tree provides a non-linear tree-based approach.

Random Forest provides an ensemble-based non-linear model.

Random Forest was subsequently tuned to improve the final model configuration.
### Model Comparison

The evaluated models produced the following held-out test results:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6783 | 0.6553 | 0.5498 | 0.5979 | 0.7317 |
| Decision Tree | 0.6492 | 0.6292 | 0.4713 | 0.5389 | 0.6630 |
| Random Forest | 0.6525 | 0.5897 | 0.6609 | 0.6233 | 0.7239 |
| Tuned Random Forest | 0.6483 | 0.5868 | 0.6475 | 0.6157 | 0.7220 |

Accuracy is considered together with precision, recall, F1-score and ROC-AUC because different error types have different implications for lead-conversion prediction.

### Confusion Matrices

The held-out test confusion matrices were:

**Logistic Regression**

```text
[[527, 151],
 [235, 287]]
 
## 8. Final Model

The final model is:

**Tuned Random Forest**

The trained model is stored as:

`models/tuned_random_forest.joblib`

The model works together with:

`models/preprocessor.joblib`

The combination ensures that incoming raw lead information is transformed consistently before being passed to the trained classifier.

---

## 9. Threshold Analysis

The model produces a probability of conversion rather than only a binary class.

Threshold analysis was performed to determine an operational classification threshold.

The analysis identified:

- Best analysis threshold: **0.42**
- Best analysis F1-score: **approximately 0.6572**

The deployment threshold was therefore set to:

**0.42**

This threshold is used by the prediction engine when converting the model probability into the final predicted class.

The threshold is configurable in:

`src/predictor.py`

---

## 10. Final Test Performance

The final Tuned Random Forest was evaluated on the held-out test dataset.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.6483 |
| Precision | 0.5868 |
| Recall | 0.6475 |
| F1-score | 0.6157 |
| ROC-AUC | 0.7220 |

### Interpretation

The model achieves a ROC-AUC of approximately **0.7220**, indicating useful discrimination between converted and non-converted leads.

The F1-score of approximately **0.6157** reflects the balance between precision and recall at the selected classification threshold.

Accuracy is not treated as the sole evaluation criterion because lead conversion is a classification problem where precision, recall, F1-score and ROC-AUC provide additional information about model behaviour.

---
## 11. Feature Importance

Feature importance analysis was performed for the final Tuned Random Forest.

Among the most influential processed features were:

| Feature | Importance |
|---|---:|
| response_time_hours | 0.1334 |
| quotation_value | 0.1195 |
| lead_age_days | 0.0876 |

Feature importance represents predictive contribution within the fitted model and should not be interpreted as causal impact.

---

## 12. Model Artifacts

The final trained artifacts are stored under the `models/` directory.

### Preprocessor

`models/preprocessor.joblib`

Contains the fitted preprocessing transformations required to convert raw input features into the 33-feature model representation.

### Final Model

`models/tuned_random_forest.joblib`

Contains the trained Tuned Random Forest classifier.

### Metadata

`models/model_metadata.json`

Stores important model information including:

- Model type
- Model artifact names
- Processed feature count
- Analysis threshold
- Best analysis F1-score
- Final test metrics

This provides a lightweight record of the final model configuration and evaluation results.

---

## 13. Prediction Engine

The reusable prediction engine is implemented in:

`src/predictor.py`

The prediction engine:

1. Receives raw lead information.
2. Validates the expected input features.
3. Separates categorical and numerical inputs.
4. Applies the saved preprocessing pipeline.
5. Generates a conversion probability.
6. Applies the deployment threshold of 0.42.
7. Produces the predicted class.
8. Assigns an operational conversion-potential category.

The prediction engine loads the saved artifacts directly and does not retrain the model during prediction.

---

## 14. Conversion Potential Categories

The model probability is converted into three operational categories.

| Probability | Category |
|---|---|
| `< 0.40` | Low Conversion Potential |
| `0.40 – < 0.70` | Medium Conversion Potential |
| `>= 0.70` | High Conversion Potential |

These categories are operational interpretations of the model probability and are separate from the underlying probability produced by the classifier.

---

## 15. FastAPI Prediction API

A FastAPI layer was added to expose the prediction engine through HTTP.

The implementation is located in:

`src/api.py`

### Available Endpoints

#### GET `/`

Provides basic API information.

#### GET `/health`

Checks whether the prediction service and model artifacts are available.

#### POST `/predict`

Accepts lead information and returns the model prediction.

The API uses Pydantic validation for incoming request data.

Validation includes constraints such as:

- Non-negative lead age
- Non-negative interactions
- Non-negative followups
- Non-negative response time
- Binary fields restricted to valid values
- Non-negative quotation value
- Non-negative website visits
- Non-negative salesperson experience

---

## 16. API Validation

The FastAPI implementation was validated using the Swagger interface.

The following cases were tested:

### Valid Prediction

A valid lead request successfully produced:

- Conversion probability
- Conversion percentage
- Predicted class
- Prediction label
- Deployment threshold
- Risk/conversion-potential category

An example verified response included approximately:

- Conversion probability: **0.5376**
- Conversion percentage: **53.76%**
- Predicted class: **1**
- Prediction: **Converted**
- Threshold: **0.42**
- Risk level: **Medium Conversion Potential**

### Health Check

The `/health` endpoint successfully confirmed model availability.

### Invalid Requests

Validation was also tested for:

- Malformed JSON
- Missing required fields
- Incorrect field types

FastAPI correctly returned HTTP `422` validation responses for invalid requests.

---

## 17. Reproducibility

The project is designed so that the trained artifacts can be reused without retraining.

The required Python dependencies are defined in:

`requirements.txt`

The project contains:

- Synthetic dataset generation
- Data validation
- Preprocessing
- Model training
- Saved model artifacts
- Prediction engine
- FastAPI API
- Documentation

The model and preprocessing artifacts are serialized using Joblib.

This allows prediction to be performed using the saved model rather than requiring model training every time the application starts.

---

## 18. Limitations

The project has several important limitations.

### Synthetic Dataset

The dataset is locally generated synthetic data rather than real production lead data.

Therefore, the model's performance should not be interpreted as production-level performance on real company data.

### Generalization

Real-world lead behaviour may differ from the relationships represented by the synthetic data.

### Model Performance

The final test F1-score is approximately 0.6157 and ROC-AUC is approximately 0.7220.

Therefore, the model should be considered a predictive prototype rather than a guaranteed conversion decision system.

### Probability Interpretation

The predicted probability is a model output and should not automatically be interpreted as a perfectly calibrated real-world probability.

### Threshold

The 0.42 threshold was selected through analysis on the project data. A production system would require validation using representative business data and an appropriate business objective.

### Feature Availability

The system assumes that the required input information is available at prediction time.

---

## 19. Conclusion

The Lead Conversion Prediction Engine implements the complete required Machine Learning workflow from synthetic data generation through prediction.

The completed system includes:

- Synthetic dataset generation
- Data-quality validation
- Exploratory Data Analysis
- Leakage analysis
- Leakage-safe preprocessing
- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest
- Model evaluation
- Threshold analysis
- Feature analysis
- Joblib model serialization
- Reusable prediction engine
- FastAPI prediction service
- Input validation
- Model metadata
- Project documentation

The final selected model is the **Tuned Random Forest**.

The final model uses **14 raw input features**, which are transformed into **33 processed features**.

The deployment threshold is **0.42**.

Final held-out test performance:

- **Accuracy:** 0.6483
- **Precision:** 0.5868
- **Recall:** 0.6475
- **F1-score:** 0.6157
- **ROC-AUC:** 0.7220

The project therefore provides a reproducible end-to-end lead-conversion prediction prototype while clearly documenting the limitations associated with synthetic data and model performance.