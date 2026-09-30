"""
Evaluate the saved lead conversion model without retraining.

Run from the project root (after python src/train.py):

    python src/evaluate.py

What it does
------------
1. Loads the saved pipeline and metadata from models/.
2. Rebuilds the same stratified 80/20 split used in training
   (same data file, same random_state), so the test set is identical.
3. Scores the untouched test set with the saved pipeline.
4. Prints accuracy, precision, recall, F1, ROC-AUC, Brier score and the
   confusion matrix at the saved decision threshold and at 0.50.
5. Prints the actual conversion rate of each potential category.
6. Checks the results against the test metrics stored in the metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import (  # noqa: E402
    DEPLOYMENT_THRESHOLD,
    conversion_category,
    metadata,
    pipeline,
)
from src.train import (  # noqa: E402
    DEFAULT_THRESHOLD,
    classification_metrics,
    load_data,
    split_data,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "brier_score")


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print("-" * 60)
    for name in METRIC_NAMES:
        print(f"  {name:<12}: {metrics[name]:.4f}")
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    print("  confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"    [[{tn:>4}, {fp:>4}],")
    print(f"     [{fn:>4}, {tp:>4}]]")
    print(f"  -> finds {tp} of {tp + fn} real buyers; {fp} false alarms")


def category_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    categories = pd.Series(
        [conversion_category(p) for p in probabilities],
        index=y_true.index,
        name="category",
    )
    table = (
        pd.DataFrame({"category": categories, "converted": y_true})
        .groupby("category")["converted"]
        .agg(leads="size", actual_conversion_rate="mean")
    )
    order = ["High Potential", "Medium Potential", "Low Potential"]
    return table.reindex([c for c in order if c in table.index])


def matches_metadata(computed: dict, stored: dict) -> bool:
    """True if every metric agrees with the value saved at training time."""
    for name in METRIC_NAMES:
        if abs(round(computed[name], 4) - stored[name]) > 1e-4:
            return False
    return computed["confusion_matrix"] == stored["confusion_matrix"]


def main() -> None:
    print("=" * 60)
    print("LEAD CONVERSION MODEL EVALUATION (saved model, no retraining)")
    print("=" * 60)

    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print(f"Model          : {metadata['model_name']}")
    print(f"Test leads     : {len(X_test)}  (conversion rate {y_test.mean():.3f})")
    print(f"Threshold      : {DEPLOYMENT_THRESHOLD:.2f}  ({metadata['threshold_method']})")

    at_threshold = classification_metrics(y_test, probabilities, DEPLOYMENT_THRESHOLD)
    at_default = classification_metrics(y_test, probabilities, DEFAULT_THRESHOLD)

    print_metrics(f"Test set at the decision threshold ({DEPLOYMENT_THRESHOLD:.2f})", at_threshold)
    print_metrics(f"Test set at the default threshold ({DEFAULT_THRESHOLD:.2f})", at_default)

    majority_accuracy = max(y_test.mean(), 1 - y_test.mean())
    print(f"\nBaseline: predicting the majority class for everyone gives "
          f"accuracy {majority_accuracy:.4f} (and recall 0).")

    print("\nActual conversion rate by potential category (test set)")
    print("-" * 60)
    print(category_table(y_test, probabilities).round(3).to_string())

    consistent = matches_metadata(
        at_threshold, metadata["test_metrics_at_decision_threshold"]
    ) and matches_metadata(at_default, metadata["test_metrics_at_0_50"])
    print("\nMatches the metrics saved at training time:", "YES" if consistent else "NO")
    if not consistent:
        print("  The data or the model changed since training. Re-run: python src/train.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
