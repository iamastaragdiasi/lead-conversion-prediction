# Project Approach Document
## Lead Conversion Prediction Engine

### 1. Problem Understanding
Build a reproducible Machine Learning system that estimates the probability that a sales lead will convert into a customer. The prediction must use information legitimately available at prediction time and must not rely on post-conversion information.

### 2. Proposed Dataset
Use a locally generated synthetic dataset containing more than 5,000 lead records. The data will model realistic business relationships among lead source, engagement, response time, quotation activity, previous-customer status, demo attendance and conversion. The dataset will contain deliberately limited, documented data-quality issues such as missing values, duplicate rows and invalid values so the cleaning process can be demonstrated honestly.

No Codly customer, confidential or proprietary data will be used.

### 3. Proposed Features
- lead_id
- lead_source
- industry
- location
- company_size
- lead_age_days
- interactions
- followups
- response_time_hours
- quotation_sent
- quotation_value
- website_visits
- previous_customer
- demo_attended
- salesperson_experience

Target: `converted` (1 = converted, 0 = not converted).

The identifier `lead_id` will not be used as a predictive feature.

### 4. Target Variable
`converted` is the binary target. The final system will expose a conversion probability and a documented business-facing prediction category.

### 5. Planned Preprocessing
1. Inspect and document data quality.
2. Remove exact duplicate records.
3. Convert identified impossible numeric values to missing values.
4. Split the cleaned feature/target data into 80% training and 20% testing using stratification.
5. Build a scikit-learn `ColumnTransformer` for numerical and categorical variables.
6. Use imputers for missing values.
7. Scale numerical variables where appropriate.
8. One-hot encode categorical variables.
9. Combine preprocessing and each classifier inside a single scikit-learn `Pipeline` to reduce leakage risk and ensure identical transformations at prediction time.

### 6. Models
Required models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

No external LLM or paid AI service is required or permitted for this task.

### 7. Evaluation Metrics
Every model will be evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The analysis will explain why accuracy alone can be misleading when the target classes are imbalanced.

### 8. Expected Workflow
Dataset generation -> data-quality validation -> EDA -> leakage analysis -> train/test split -> preprocessing pipeline -> model training -> model comparison -> feature-importance analysis -> final model selection -> Joblib serialization -> local prediction program -> report and README -> Git submission.

### 9. Potential Challenges
- Class imbalance
- Missing values
- Duplicate and invalid records
- Legitimate extreme values versus erroneous outliers
- Data leakage
- Overfitting
- Differences between training and unseen test data
- Interpreting feature importance without treating it as causality
- Choosing and documenting probability-category thresholds

### 10. Proposed Timeline
Phase 1: Project setup and approach document
Phase 2: Synthetic dataset generation and validation
Phase 3: Complete EDA
Phase 4: Leakage-safe preprocessing
Phase 5: Required model training
Phase 6: Evaluation and comparison
Phase 7: Feature importance and final model
Phase 8: Joblib prediction application
Phase 9: Documentation, Git history and final audit
