"""
Reusable prediction engine.

Loads the saved preprocessing + model pipeline and its metadata once, and
scores new leads without retraining. Used by the CLI (src/predict.py) and the
FastAPI service (src/api.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# ============================================================
# ARTIFACT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "lead_conversion_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ============================================================
# LOAD ARTIFACTS (once, at import time)
# ============================================================

if not PIPELINE_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError(
        "Model artifacts not found. Train the model first:  python src/train.py"
    )

pipeline = joblib.load(PIPELINE_PATH)
metadata: dict[str, Any] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

CATEGORICAL_FEATURES: list[str] = metadata["categorical_features"]
NUMERICAL_FEATURES: list[str] = metadata["numerical_features"]
FEATURES: list[str] = metadata["features"]

# Chosen during training from out-of-fold predictions on the training set.
DEPLOYMENT_THRESHOLD: float = float(metadata["decision_threshold"])
HIGH_POTENTIAL_CUTOFF: float = float(metadata["high_potential_cutoff"])


# ============================================================
# INPUT LIMITS
# ============================================================

# Plausible business ranges used to reject typos and impossible values.
# They are deliberately wider than the training data; they are not model limits.
NUMERIC_LIMITS: dict[str, tuple[float, float]] = {
    "lead_age_days": (0, 1825),              # up to 5 years
    "interactions": (0, 100),
    "followups": (0, 50),
    "response_time_hours": (0, 720),         # up to 30 days
    "quotation_sent": (0, 1),
    "quotation_value": (0, 10_000_000),      # up to INR 1 crore
    "website_visits": (0, 200),
    "previous_customer": (0, 1),
    "demo_attended": (0, 1),
    "salesperson_experience": (1, 50),       # same rule as validate_data.py
}

WHOLE_NUMBER_FEATURES = {"lead_age_days", "interactions", "followups", "website_visits"}
BINARY_FEATURES = ("quotation_sent", "previous_customer", "demo_attended")


# ============================================================
# HELPERS
# ============================================================

def conversion_category(probability: float) -> str:
    """
    Business-facing category.

    Low / Medium boundary = the analysed decision threshold.
    Medium / High boundary (0.70) = a business convention, not a
    statistically derived value.
    """
    if probability >= HIGH_POTENTIAL_CUTOFF:
        return "High Potential"
    if probability >= DEPLOYMENT_THRESHOLD:
        return "Medium Potential"
    return "Low Potential"


def _known_categories() -> dict[str, set[str]]:
    encoder = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )
    return {
        feature: {str(v) for v in values}
        for feature, values in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }


KNOWN_CATEGORIES = _known_categories()


def _validate(lead_data: dict[str, Any]) -> None:
    missing = [f for f in FEATURES if f not in lead_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    for feature in CATEGORICAL_FEATURES:
        value = lead_data[feature]
        if value is not None and str(value) not in KNOWN_CATEGORIES[feature]:
            allowed = sorted(KNOWN_CATEGORIES[feature])
            raise ValueError(f"'{feature}' must be one of {allowed}, got {value!r}")

    for feature in NUMERICAL_FEATURES:
        value = lead_data[feature]
        if value is None:
            continue  # the pipeline imputes missing numeric values
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{feature}' must be numeric, got {value!r}") from None
        if math.isnan(number):
            continue  # treated as missing and imputed by the pipeline
        if math.isinf(number):
            raise ValueError(f"'{feature}' must be a finite number, got {value!r}")
        low, high = NUMERIC_LIMITS[feature]
        if not low <= number <= high:
            raise ValueError(f"'{feature}' must be between {low:g} and {high:g}, got {number:g}")
        if feature in WHOLE_NUMBER_FEATURES and not number.is_integer():
            raise ValueError(f"'{feature}' must be a whole number, got {number:g}")

    # Yes/no flags must be exactly 0 or 1 (0.5 is not a valid answer).
    for feature in BINARY_FEATURES:
        value = lead_data[feature]
        if value is not None and float(value) not in (0.0, 1.0):
            raise ValueError(f"'{feature}' must be 0 or 1, got {value!r}")

    # A quotation value only makes sense if a quotation was sent.
    sent = lead_data["quotation_sent"]
    quote = lead_data["quotation_value"]
    if sent is not None and quote is not None and float(sent) == 0 and float(quote) > 0:
        raise ValueError(
            "'quotation_value' must be 0 when 'quotation_sent' is 0, "
            f"got {float(quote):g}"
        )


# ============================================================
# PREDICTION
# ============================================================

def predict_lead(lead_data: dict[str, Any]) -> dict[str, Any]:
    """
    Predict the conversion probability of a single lead.

    Returns the probability, percentage, predicted class, label, the
    threshold used and the conversion-potential category.
    """
    _validate(lead_data)

    input_frame = pd.DataFrame([lead_data], columns=FEATURES)
    probability = float(pipeline.predict_proba(input_frame)[0, 1])
    predicted_class = int(probability >= DEPLOYMENT_THRESHOLD)
    category = conversion_category(probability)

    return {
        "conversion_probability": round(probability, 4),
        "conversion_percentage": round(probability * 100, 2),
        "predicted_class": predicted_class,
        "prediction": "Likely to Convert" if predicted_class else "Unlikely to Convert",
        "threshold": DEPLOYMENT_THRESHOLD,
        "category": category,
    }


def predict_leads(leads: pd.DataFrame) -> pd.DataFrame:
    """Score many leads at once (e.g. a CSV export from a CRM)."""
    missing = [f for f in FEATURES if f not in leads.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for row_number, row in enumerate(leads[FEATURES].to_dict(orient="records"), start=1):
        clean_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        try:
            _validate(clean_row)
        except ValueError as exc:
            raise ValueError(f"Row {row_number}: {exc}") from None

    probabilities = pipeline.predict_proba(leads[FEATURES])[:, 1]
    scored = leads.copy()
    scored["conversion_probability"] = probabilities.round(4)
    scored["predicted_class"] = (probabilities >= DEPLOYMENT_THRESHOLD).astype(int)
    scored["category"] = [conversion_category(p) for p in probabilities]
    return scored


def model_health() -> dict[str, Any]:
    """Basic information about the loaded model."""
    return {
        "model_loaded": pipeline is not None,
        "model_name": metadata["model_name"],
        "model_type": type(pipeline.named_steps["model"]).__name__,
        "feature_count": len(FEATURES),
        "deployment_threshold": DEPLOYMENT_THRESHOLD,
        "threshold_method": metadata["threshold_method"],
    }
