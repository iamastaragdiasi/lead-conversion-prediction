from pathlib import Path
from typing import Any

import joblib
import pandas as pd


# ============================================================
# PROJECT / MODEL ARTIFACT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
MODEL_PATH = MODEL_DIR / "tuned_random_forest.joblib"


# ============================================================
# DEPLOYMENT CONFIGURATION
# ============================================================

# Final threshold identified during model analysis.
# Best F1 was achieved at approximately 0.40.
DEPLOYMENT_THRESHOLD = 0.40


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

preprocessor = joblib.load(PREPROCESSOR_PATH)

model = joblib.load(MODEL_PATH)


# ============================================================
# REQUIRED FEATURES
# ============================================================

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

FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_lead(lead_data: dict[str, Any]) -> dict[str, Any]:
    """
    Predict whether a lead will convert.

    Parameters
    ----------
    lead_data:
        Dictionary containing the required lead features.

    Returns
    -------
    dict
        Prediction result containing:
        - conversion probability
        - conversion percentage
        - predicted class
        - prediction label
        - threshold
        - risk level
    """

    # --------------------------------------------------------
    # Validate required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in lead_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [lead_data],
        columns=FEATURES
    )

    # --------------------------------------------------------
    # Apply the SAME preprocessing used during training
    # --------------------------------------------------------

    processed_data = preprocessor.transform(
        input_data
    )

    # --------------------------------------------------------
    # Generate conversion probability
    # --------------------------------------------------------

    conversion_probability = float(
        model.predict_proba(processed_data)[0][1]
    )

    # --------------------------------------------------------
    # Classification threshold
    # --------------------------------------------------------

    threshold = DEPLOYMENT_THRESHOLD

    predicted_class = int(
        conversion_probability >= threshold
    )

    # --------------------------------------------------------
    # Human-readable prediction
    # --------------------------------------------------------

    if predicted_class == 1:
        prediction = "Converted"
    else:
        prediction = "Not Converted"

    # --------------------------------------------------------
    # Risk / opportunity level
    # --------------------------------------------------------

    if conversion_probability >= 0.70:
        risk_level = "High Conversion Potential"

    elif conversion_probability >= 0.40:
        risk_level = "Medium Conversion Potential"

    else:
        risk_level = "Low Conversion Potential"

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "conversion_probability": round(
            conversion_probability,
            4
        ),

        "conversion_percentage": round(
            conversion_probability * 100,
            2
        ),

        "predicted_class": predicted_class,

        "prediction": prediction,

        "threshold": threshold,

        "risk_level": risk_level,
    }


# ============================================================
# MODEL HEALTH CHECK
# ============================================================

def model_health() -> dict[str, Any]:
    """
    Return basic information about the loaded model.
    """

    return {
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": type(model).__name__,
        "feature_count": len(FEATURES),
        "deployment_threshold": DEPLOYMENT_THRESHOLD,
    }