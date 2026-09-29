"""
Train, compare and save the lead conversion model.

Run from the project root:

    python src/train.py

Workflow
--------
1. Load the validated dataset (data/processed/leads_validated.csv).
2. Stratified 80/20 train/test split (random_state=42).
3. Build one scikit-learn Pipeline per model: preprocessing + classifier.
   Preprocessing is inside the pipeline, so during cross-validation it is
   re-fitted on each training fold and never sees the validation fold.
4. Compare Logistic Regression, Decision Tree, Random Forest and a
   GridSearchCV-tuned Random Forest with 5-fold cross-validation on the
   TRAINING set only.
5. Select the final model by cross-validated ROC-AUC (training data only).
6. Choose the decision threshold from out-of-fold predictions on the
   training set (maximum F1). The test set is not used for any decision.
7. Evaluate every model once on the untouched test set.
8. Save the complete pipeline, metadata, metrics and figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.50
HIGH_POTENTIAL_CUTOFF = 0.70

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "leads_validated.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PIPELINE_PATH = MODEL_DIR / "lead_conversion_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET = "converted"
ID_COLUMN = "lead_id"

CATEGORICAL_FEATURES = [
    "lead_source",
    "industry",
    "location",
    "company_size",
]

NUMERICAL_FEATURES = [
    "lead_age_days",
    "interactions",
    "followups",
    "response_time_hours",
    "quotation_sent",
    "quotation_value",
    "website_visits",
    "previous_customer",
    "demo_attended",
    "salesperson_experience",
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "tuned_random_forest": "Tuned Random Forest",
}

RF_PARAM_GRID = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [6, 10, 14],
    "model__min_samples_split": [10, 20],
    "model__min_samples_leaf": [5, 10],
}


# ============================================================
# DATA
# ============================================================

def load_data(path: Path = DATA_FILE) -> tuple[pd.DataFrame, pd.Series]:
    """Load the validated dataset and separate features from the target."""
    df = pd.read_csv(path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # lead_id is an identifier, not a predictive signal, so it is excluded.
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 split so both sets keep the same conversion rate."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ============================================================
# PIPELINES
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """Imputation + scaling for numbers, imputation + one-hot for categories."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    """The three required baseline models (same settings as the notebook)."""
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )
        ),
        "decision_tree": build_pipeline(
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def tune_random_forest(X_train, y_train) -> GridSearchCV:
    """Grid search on the training set only (preprocessing inside each fold)."""
    search = GridSearchCV(
        estimator=build_pipeline(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        param_grid=RF_PARAM_GRID,
        scoring="roc_auc",
        cv=cv_splitter(),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


# ============================================================
# EVALUATION
# ============================================================

def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "threshold": round(float(threshold), 2),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def out_of_fold_probabilities(pipeline, X_train, y_train) -> np.ndarray:
    """Probabilities for each training row from a model that never saw it."""
    return cross_val_predict(
        clone(pipeline),
        X_train,
        y_train,
        cv=cv_splitter(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


def threshold_table(y_true, probabilities) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.10, 0.91, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(y_train, oof_probabilities) -> tuple[float, pd.DataFrame]:
    """Maximum-F1 threshold on out-of-fold TRAINING predictions only."""
    table = threshold_table(y_train, oof_probabilities)
    best_row = table.loc[table["f1"].idxmax()]
    return round(float(best_row["threshold"]), 2), table


def conversion_category(probability: float, threshold: float) -> str:
    if probability >= HIGH_POTENTIAL_CUTOFF:
        return "High Potential"
    if probability >= threshold:
        return "Medium Potential"
    return "Low Potential"


def run_experiment(X_train, X_test, y_train, y_test, verbose: bool = True) -> dict:
    """Fit all models, compare them on training CV and evaluate on the test set."""
    log = print if verbose else (lambda *args, **kwargs: None)

    pipelines = candidate_models()

    log("Tuning Random Forest with GridSearchCV (training data only)...")
    search = tune_random_forest(X_train, y_train)
    pipelines["tuned_random_forest"] = search.best_estimator_
    best_params = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }
    log(f"  Best parameters: {best_params}")

    results = {}
    for name, pipeline in pipelines.items():
        log(f"Evaluating {MODEL_LABELS[name]}...")

        oof = out_of_fold_probabilities(pipeline, X_train, y_train)

        fitted = clone(pipeline).fit(X_train, y_train)
        train_probabilities = fitted.predict_proba(X_train)[:, 1]
        test_probabilities = fitted.predict_proba(X_test)[:, 1]

        cv_threshold, _ = select_threshold(y_train, oof)

        results[name] = {
            "label": MODEL_LABELS[name],
            "pipeline": fitted,
            "oof_probabilities": oof,
            "test_probabilities": test_probabilities,
            "train_roc_auc": float(roc_auc_score(y_train, train_probabilities)),
            "cv_roc_auc": float(roc_auc_score(y_train, oof)),
            "cv_threshold": cv_threshold,
            "test_default": classification_metrics(
                y_test, test_probabilities, DEFAULT_THRESHOLD
            ),
            "test_cv_threshold": classification_metrics(
                y_test, test_probabilities, cv_threshold
            ),
        }

    final_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    threshold, table = select_threshold(
        y_train, results[final_name]["oof_probabilities"]
    )

    return {
        "results": results,
        "final_name": final_name,
        "threshold": threshold,
        "threshold_table": table,
        "rf_best_params": best_params,
        "rf_best_cv_roc_auc": float(search.best_score_),
    }


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        m = r["test_default"]
        rows.append(
            {
                "Model": r["label"],
                "Train ROC-AUC": r["train_roc_auc"],
                "CV ROC-AUC": r["cv_roc_auc"],
                "Test Accuracy": m["accuracy"],
                "Test Precision": m["precision"],
                "Test Recall": m["recall"],
                "Test F1": m["f1"],
                "Test ROC-AUC": m["roc_auc"],
                "Test Brier": m["brier_score"],
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def feature_importance(final_pipeline, X_test, y_test) -> pd.DataFrame:
    """Permutation importance on the held-out test set (raw features)."""
    result = permutation_importance(
        final_pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def logistic_coefficients(pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    return (
        pd.DataFrame({"feature": names, "coefficient": coefficients})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )


def rf_impurity_importance(pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# FIGURES
# ============================================================

def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_figures(experiment: dict, y_train, y_test, importance, coefficients, rf_importance):
    results = experiment["results"]
    final = results[experiment["final_name"]]

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results.values():
        fpr, tpr, _ = roc_curve(y_test, r["test_probabilities"])
        ax.plot(fpr, tpr, label=f"{r['label']} (AUC {r['test_default']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    ax.set(title="ROC Curves — Test Set", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend()
    _save(fig, "roc_curve_comparison.png")

    # Precision-recall curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results.values():
        precision, recall, _ = precision_recall_curve(y_test, r["test_probabilities"])
        ax.plot(recall, precision, label=r["label"])
    ax.axhline(y_test.mean(), linestyle="--", color="grey", label="Base rate")
    ax.set(title="Precision-Recall Curves — Test Set", xlabel="Recall", ylabel="Precision")
    ax.legend()
    _save(fig, "precision_recall_curves.png")

    # Metric comparison at the default threshold
    table = comparison_table(results)[
        ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC-AUC"]
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    table.plot(kind="bar", ax=ax, rot=0)
    ax.set(title="Model Comparison — Test Set (threshold 0.50)", ylabel="Score", ylim=(0, 1))
    ax.legend(title="Metric", loc="lower right")
    _save(fig, "model_performance_comparison.png")

    # Confusion matrices at the default threshold
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4))
    for ax, r in zip(axes, results.values()):
        cm = np.array(r["test_default"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for (i, j), value in np.ndenumerate(cm):
            shade = (value - cm.min()) / max(cm.max() - cm.min(), 1)
            color = "white" if shade > 0.6 else "black"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13,
                    color=color, fontweight="bold")
        ax.set(
            title=r["label"],
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"],
        )
    fig.suptitle("Confusion Matrices — Test Set (threshold 0.50)")
    _save(fig, "confusion_matrices.png")

    # Threshold analysis (training out-of-fold predictions)
    t = experiment["threshold_table"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t["threshold"], t["precision"], label="Precision")
    ax.plot(t["threshold"], t["recall"], label="Recall")
    ax.plot(t["threshold"], t["f1"], label="F1-score")
    ax.axvline(experiment["threshold"], linestyle="--", color="grey",
               label=f"Selected threshold {experiment['threshold']:.2f}")
    ax.set(
        title=f"Threshold Analysis — {final['label']} (5-fold out-of-fold, training set only)",
        xlabel="Decision threshold",
        ylabel="Score",
    )
    ax.legend()
    _save(fig, "threshold_analysis.png")

    # Permutation importance
    top = importance.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set(
        title=f"Permutation Importance — {final['label']} (drop in test ROC-AUC)",
        xlabel="Mean decrease in ROC-AUC",
    )
    _save(fig, "permutation_importance.png")

    # Logistic Regression coefficients
    top = coefficients.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["tab:green" if c > 0 else "tab:red" for c in top["coefficient"]]
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Logistic Regression Coefficients (standardised / one-hot, top 15)",
           xlabel="Coefficient (log-odds)")
    _save(fig, "logistic_regression_coefficients.png")

    # Random Forest impurity importance (for comparison)
    top = rf_importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set(title="Tuned Random Forest — Impurity Importance (top 15)", xlabel="Importance")
    _save(fig, "random_forest_feature_importance.png")


# ============================================================
# SAVE
# ============================================================

def _round(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def save_outputs(experiment: dict, importance: pd.DataFrame, X_train, X_test) -> dict:
    results = experiment["results"]
    final_name = experiment["final_name"]
    final = results[final_name]
    threshold = experiment["threshold"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final["pipeline"], PIPELINE_PATH)

    metadata = {
        "model_name": final["label"],
        "model_key": final_name,
        "pipeline_file": PIPELINE_PATH.name,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "decision_threshold": threshold,
        "threshold_method": (
            "Maximum F1 on 5-fold out-of-fold predictions from the training set only"
        ),
        "categories": {
            "Low Potential": f"probability < {threshold:.2f}",
            "Medium Potential": f"{threshold:.2f} <= probability < {HIGH_POTENTIAL_CUTOFF:.2f}",
            "High Potential": f"probability >= {HIGH_POTENTIAL_CUTOFF:.2f}",
        },
        "high_potential_cutoff": HIGH_POTENTIAL_CUTOFF,
        "selection_criterion": "Highest 5-fold cross-validated ROC-AUC on the training set",
        "cv_roc_auc": final["cv_roc_auc"],
        "test_metrics_at_decision_threshold": final["test_cv_threshold"],
        "test_metrics_at_0_50": final["test_default"],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
    }
    METADATA_PATH.write_text(json.dumps(_round(metadata), indent=4) + "\n", encoding="utf-8")

    metrics = {
        "final_model": final["label"],
        "decision_threshold": threshold,
        "tuned_random_forest_best_params": experiment["rf_best_params"],
        "tuned_random_forest_grid_cv_roc_auc": experiment["rf_best_cv_roc_auc"],
        "models": {
            name: {
                "label": r["label"],
                "train_roc_auc": r["train_roc_auc"],
                "cv_roc_auc": r["cv_roc_auc"],
                "cv_threshold": r["cv_threshold"],
                "test_at_0_50": r["test_default"],
                "test_at_own_cv_threshold": r["test_cv_threshold"],
            }
            for name, r in results.items()
        },
        "permutation_importance": importance.to_dict(orient="records"),
    }
    METRICS_PATH.write_text(json.dumps(_round(metrics), indent=4) + "\n", encoding="utf-8")
    return metadata


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("LEAD CONVERSION MODEL TRAINING")
    print("=" * 70)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Rows: {len(X)}  |  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Conversion rate — train: {y_train.mean():.3f}  test: {y_test.mean():.3f}\n")

    experiment = run_experiment(X_train, X_test, y_train, y_test)
    results = experiment["results"]
    final = results[experiment["final_name"]]

    print("\nModel comparison (test metrics at threshold 0.50):")
    print(comparison_table(results).round(4).to_string())

    print(f"\nSelected model : {final['label']} (highest CV ROC-AUC {final['cv_roc_auc']:.4f})")
    print(f"Threshold      : {experiment['threshold']:.2f} (max F1, out-of-fold, training set)")

    m = final["test_cv_threshold"]
    print("\nFinal model — test set at the decision threshold:")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {key:<10}: {m[key]:.4f}")
    print(f"  confusion : {m['confusion_matrix']}  ([[TN, FP], [FN, TP]])")

    print("\nComputing permutation importance...")
    importance = feature_importance(final["pipeline"], X_test, y_test)
    print(importance.round(4).to_string(index=False))

    coefficients = logistic_coefficients(results["logistic_regression"]["pipeline"])
    rf_importance = rf_impurity_importance(results["tuned_random_forest"]["pipeline"])

    save_figures(experiment, y_train, y_test, importance, coefficients, rf_importance)
    save_outputs(experiment, importance, X_train, X_test)

    print(f"\nSaved pipeline : {PIPELINE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved metadata : {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics  : {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved figures  : {FIGURES_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
