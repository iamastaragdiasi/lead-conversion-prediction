"""
Command-line lead conversion predictor.

Loads the saved pipeline (no retraining) and predicts a new lead.

Usage (from the project root):

    python src/predict.py                      # interactive prompts
    python src/predict.py --example            # score a built-in example lead
    python src/predict.py --input lead.json    # score a lead stored as JSON
    python src/predict.py --csv leads.csv      # score many leads, write *_scored.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    metadata,
    pipeline,
    predict_lead,
    predict_leads,
)

EXAMPLE_LEAD = {
    "lead_source": "Website",
    "industry": "Retail",
    "location": "Bengaluru",
    "company_size": "Medium",
    "lead_age_days": 20,
    "interactions": 6,
    "followups": 3,
    "response_time_hours": 2.0,
    "quotation_sent": 1,
    "quotation_value": 85000,
    "website_visits": 5,
    "previous_customer": 0,
    "demo_attended": 1,
    "salesperson_experience": 6,
}

PROMPTS = {
    "lead_source": "Lead source",
    "industry": "Industry",
    "location": "Location",
    "company_size": "Company size",
    "lead_age_days": "Lead age (days)",
    "interactions": "Interactions",
    "followups": "Follow-ups",
    "response_time_hours": "First response time (hours)",
    "quotation_sent": "Quotation sent? (y/n)",
    "quotation_value": "Quotation value (INR, 0 if none)",
    "website_visits": "Website visits",
    "previous_customer": "Previous customer? (y/n)",
    "demo_attended": "Demo attended? (y/n)",
    "salesperson_experience": "Salesperson experience (years)",
}

YES_NO_FEATURES = {"quotation_sent", "previous_customer", "demo_attended"}
INTEGER_FEATURES = {
    "lead_age_days",
    "interactions",
    "followups",
    "website_visits",
}


def known_categories() -> dict[str, list[str]]:
    """Category values the model was trained on (read from the saved encoder)."""
    encoder = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )
    return {
        feature: [str(v) for v in values]
        for feature, values in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }


def ask(feature: str, choices: dict[str, list[str]]):
    label = PROMPTS[feature]
    while True:
        if feature in choices:
            options = choices[feature]
            print(f"\n{label}:")
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option}")
            raw = input("Choose number: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
        else:
            raw = input(f"{label}: ").strip().lower()
            if feature in YES_NO_FEATURES:
                if raw in ("y", "yes", "1"):
                    return 1
                if raw in ("n", "no", "0"):
                    return 0
            else:
                try:
                    value = float(raw.replace(",", ""))
                    if value >= 0:
                        return int(value) if feature in INTEGER_FEATURES else value
                except ValueError:
                    pass
        print("  Invalid input, please try again.")


def print_result(lead: dict, result: dict) -> None:
    line = "=" * 44
    print(f"\n{line}")
    print("       LEAD CONVERSION PREDICTOR")
    print(line)
    for feature in FEATURES:
        value = lead[feature]
        if feature in YES_NO_FEATURES:
            value = "Yes" if int(value) == 1 else "No"
        print(f"{PROMPTS[feature].split(' (')[0].split('?')[0]:<24}: {value}")
    print("-" * 44)
    print(f"Conversion Probability  : {result['conversion_percentage']:.1f}%")
    print(f"Prediction              : {result['prediction']}")
    print(f"Category                : {result['category'].upper()}")
    print(f"Decision threshold      : {result['threshold']:.2f}")
    print(f"Model                   : {metadata['model_name']}")
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict lead conversion probability.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--example", action="store_true", help="score a built-in example lead")
    group.add_argument("--input", type=Path, help="JSON file containing one lead")
    group.add_argument("--csv", type=Path, help="CSV file containing many leads")
    args = parser.parse_args()

    if args.csv:
        leads = pd.read_csv(args.csv)
        scored = predict_leads(leads)
        output = args.csv.with_name(f"{args.csv.stem}_scored.csv")
        scored.to_csv(output, index=False)
        print(f"Scored {len(scored)} leads -> {output}")
        print(scored["category"].value_counts().to_string())
        return

    if args.example:
        lead = EXAMPLE_LEAD
    elif args.input:
        lead = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        choices = known_categories()
        print("Enter the lead details (press Ctrl+C to quit).")
        lead = {feature: ask(feature, choices) for feature in FEATURES}

    try:
        result = predict_lead(lead)
    except ValueError as exc:
        sys.exit(f"Invalid lead: {exc}")

    print_result(lead, result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
