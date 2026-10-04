# Evaluation Report: Addendum

This addendum adds evidence for claims made in `evaluation_report.md`. None of the published model metrics changed.

## 1. Leakage: evidence, not just assertion
`reports/leakage_demo.md` contains three things:
- A controlled experiment: a planted post-outcome column (`invoice_generated`) inflates CV ROC-AUC to near-perfect.
- A single-feature screen that flags it.
- A per-feature table showing when each feature becomes known.

The synthetic generator creates every feature before the outcome, so nothing leaks by construction. The real risk appears when real CRM exports contain values recorded after the deal closes. The availability table identifies which features need point-in-time snapshots.

## 2. Calibration
`reports/calibration_analysis.md` reports Brier score, Brier skill score, expected calibration error and reliability curves for the final model, on both out-of-fold training predictions and the test set. Together these answer "what does a 78% prediction mean?" with measured evidence.

## 3. Why the conversion rate is 43.8%
The model scores leads **mid-funnel**, after first sales activity. These are engaged leads, comparable to sales-qualified leads, not raw inbound enquiries, and conversion at that stage is far higher than for raw leads.

The method does not depend on this prevalence. At a much lower conversion rate, accuracy becomes even less informative, and PR-AUC, recall and precision become the primary metrics. `class_weight="balanced"` would also be evaluated in that case.

## 4. The threshold is a business decision
0.32 maximises F1. That choice implicitly assumes how much more a missed buyer costs than a wasted call. The cost table in `calibration_analysis.md` shows the threshold implied by different cost ratios. In production the ratio should come from actual deal value and sales capacity.

## 5. Known issue: `quotation_value` imputation
Most leads have no quotation, so the median of `quotation_value` is 0. As a result, the 21 leads with a sent quotation but a missing value are imputed as 0.

The fix is a missing-value indicator (`SimpleImputer(add_indicator=True)`) or imputation conditional on `quotation_sent == 1`. It affects 21 of 6,000 leads (0.35%). It was not applied, so that the published, reviewed metrics remain unchanged.

## 6. Deviations from the approach document
- `company_size` is stored as a band rather than a head-count, because CRMs usually capture it as a range and banding reduces data-entry error.
- The decision threshold is selected from out-of-fold training predictions instead of the default 0.50.
- A GridSearchCV-tuned Random Forest was added to test whether tuning closed the gap with Logistic Regression. It did not.
- An optional FastAPI service was added as an integration example. It is isolated in `requirements-api.txt` and is not needed for the core task.
- Analysis scripts, tests, a model hash and a dependency lock file were added.

## 7. Model file security
Joblib files are pickles: **loading one executes code**. Only load model files from a trusted source. `python src/hash_model.py --check` verifies that the saved model has not been altered. A production system would keep models in an access-controlled model registry.

## 8. Reproducibility
`requirements.lock.txt` pins the exact versions used to produce the results. `requirements.txt` stays as the human-readable dependency list.
