# Shared helpers for the review/analysis scripts (leakage demo, calibration).
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "leads_validated.csv"
MODEL_PATH = ROOT / "models" / "lead_conversion_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TARGET = "converted"

# These MUST match src/train.py. calibration_analysis.py verifies this by
# reproducing the published test ROC-AUC and stops if it does not match.
TEST_SIZE = 0.20
RANDOM_STATE = 42
EXPECTED_TEST_AUC = 0.7492

FALLBACK_FEATURES = [
    "lead_source", "industry", "location", "company_size", "lead_age_days",
    "interactions", "followups", "response_time_hours", "quotation_sent",
    "quotation_value", "website_visits", "previous_customer", "demo_attended",
    "salesperson_experience",
]


def load_metadata():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


def _features_from_meta(meta):
    for key in ("features", "feature_columns", "input_features", "raw_features"):
        value = meta.get(key)
        if isinstance(value, list) and value and all(isinstance(c, str) for c in value):
            return list(value)
        if isinstance(value, dict):
            cols = [c for v in value.values() if isinstance(v, list) for c in v if isinstance(c, str)]
            if cols:
                return cols
    return None


def decision_threshold(meta):
    for key in ("threshold", "decision_threshold", "selected_threshold"):
        value = meta.get(key)
        if isinstance(value, (int, float)) and 0 < value < 1:
            return float(value)
    return 0.32


def load_split(meta):
    df = pd.read_csv(DATA_PATH)
    features = _features_from_meta(meta)
    if not features or any(c not in df.columns for c in features):
        features = [c for c in FALLBACK_FEATURES if c in df.columns]
    X = df[features]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return features, X_train, X_test, y_train, y_test
