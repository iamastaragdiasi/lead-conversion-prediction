"""
Reusable prediction engine.

Loads the saved preprocessing + model pipeline and its metadata once, and
scores new leads without retraining. Used by the CLI (src/predict.py) and the
FastAPI service (src/api.py).
"""

from __future__ import annotations

import json
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


def _validate(lead_data: dict[str, Any]) -> None:
    missing = [f for f in FEATURES if f not in lead_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    for feature in NUMERICAL_FEATURES:
        value = lead_data[feature]
        if value is None:
            continue  # the pipeline imputes missing numeric values
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{feature}' must be numeric, got {value!r}") from None
        if number < 0:
            raise ValueError(f"'{feature}' cannot be negative, got {number}")

    for feature in ("quotation_sent", "previous_customer", "demo_attended"):
        value = lead_data[feature]
        if value is not None and int(value) not in (0, 1):
            raise ValueError(f"'{feature}' must be 0 or 1, got {value!r}")


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
        # Kept for backward compatibility with earlier API clients.
        "risk_level": category,
    }


def predict_leads(leads: pd.DataFrame) -> pd.DataFrame:
    """Score many leads at once (e.g. a CSV export from a CRM)."""
    missing = [f for f in FEATURES if f not in leads.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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
