import json
from pathlib import Path

import joblib

from src.predictor import predict_lead

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "lead_conversion_pipeline.joblib"
META = ROOT / "models" / "model_metadata.json"

EXAMPLE = {
    "lead_source": "Website", "industry": "Retail", "location": "Bengaluru",
    "company_size": "Medium", "lead_age_days": 20, "interactions": 6, "followups": 3,
    "response_time_hours": 2.0, "quotation_sent": 1, "quotation_value": 85000,
    "website_visits": 5, "previous_customer": 0, "demo_attended": 1,
    "salesperson_experience": 6,
}
CLEAN_ERRORS = (ValueError, KeyError, TypeError)


def _prob(result):
    for key in ("probability", "conversion_probability", "proba"):
        if key in result:
            return float(result[key])
    raise AssertionError(f"No probability key in result: {sorted(result)}")


def _category(result):
    for key in ("category", "prediction_category", "potential"):
        if key in result:
            return str(result[key]).upper()
    raise AssertionError(f"No category key in result: {sorted(result)}")


def test_saved_pipeline_loads():
    assert hasattr(joblib.load(MODEL), "predict_proba")


def test_metadata_exists_and_is_json():
    assert isinstance(json.loads(META.read_text(encoding="utf-8")), dict)


def test_probability_in_range():
    assert 0.0 <= _prob(predict_lead(dict(EXAMPLE))) <= 1.0


def test_prediction_is_deterministic():
    assert _prob(predict_lead(dict(EXAMPLE))) == _prob(predict_lead(dict(EXAMPLE)))


def test_category_matches_documented_bands():
    result = predict_lead(dict(EXAMPLE))
    p, cat = _prob(result), _category(result)
    expected = "LOW" if p < 0.32 else ("HIGH" if p >= 0.70 else "MEDIUM")
    assert expected in cat


def test_demo_and_quotation_raise_probability():
    weak = dict(EXAMPLE, demo_attended=0, quotation_sent=0, quotation_value=0)
    strong = dict(EXAMPLE, demo_attended=1, quotation_sent=1)
    assert _prob(predict_lead(strong)) > _prob(predict_lead(weak))


def test_unknown_category_handled_cleanly():
    try:
        result = predict_lead(dict(EXAMPLE, industry="Unseen Industry"))
    except CLEAN_ERRORS:
        return
    assert 0.0 <= _prob(result) <= 1.0


def test_negative_value_handled_cleanly():
    try:
        result = predict_lead(dict(EXAMPLE, response_time_hours=-5))
    except CLEAN_ERRORS:
        return
    assert 0.0 <= _prob(result) <= 1.0
