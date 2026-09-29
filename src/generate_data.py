from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_RECORDS = 6000

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "leads_raw.csv"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n_records=N_RECORDS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    lead_id = np.arange(100001, 100001 + n_records)

    lead_source = rng.choice(
        [
            "Website",
            "Referral",
            "Email Campaign",
            "Social Media",
            "Cold Call",
            "Partner",
        ],
        size=n_records,
        p=[0.25, 0.18, 0.15, 0.14, 0.16, 0.12],
    )

    industry = rng.choice(
        [
            "Technology",
            "Finance",
            "Healthcare",
            "Retail",
            "Manufacturing",
            "Education",
        ],
        size=n_records,
        p=[0.24, 0.17, 0.15, 0.17, 0.15, 0.12],
    )

    location = rng.choice(
        [
            "Kolkata",
            "Bengaluru",
            "Hyderabad",
            "Mumbai",
            "Delhi",
            "Chennai",
            "Pune",
        ],
        size=n_records,
    )

    company_size = rng.choice(
        ["Small", "Medium", "Large", "Enterprise"],
        size=n_records,
        p=[0.30, 0.35, 0.25, 0.10],
    )

    lead_age_days = rng.integers(1, 121, size=n_records)

    interactions = np.clip(
        rng.poisson(lam=5, size=n_records),
        0,
        30,
    )

    followups = np.clip(
        rng.poisson(lam=2, size=n_records),
        0,
        12,
    )

    response_time_hours = rng.lognormal(
        mean=np.log(18),
        sigma=0.7,
        size=n_records,
    )
    response_time_hours = np.clip(
        response_time_hours,
        0.5,
        168,
    )

    quotation_sent = rng.binomial(
        1,
        0.42,
        size=n_records,
    )

    quotation_value = np.where(
        quotation_sent == 1,
        rng.lognormal(
            mean=np.log(75000),
            sigma=0.8,
            size=n_records,
        ),
        0,
    )

    quotation_value = np.clip(
        quotation_value,
        0,
        1_000_000,
    )

    website_visits = np.clip(
        rng.poisson(lam=4, size=n_records),
        0,
        40,
    )

    previous_customer = rng.binomial(
        1,
        0.18,
        size=n_records,
    )

    demo_attended = rng.binomial(
        1,
        0.35,
        size=n_records,
    )

    salesperson_experience = rng.integers(
        1,
        16,
        size=n_records,
    )

    # ---------------------------------------------------------
    # Probabilistic conversion generation
    # ---------------------------------------------------------

    source_effect = {
        "Website": 0.35,
        "Referral": 0.85,
        "Email Campaign": 0.10,
        "Social Media": -0.15,
        "Cold Call": -0.40,
        "Partner": 0.55,
    }

    industry_effect = {
        "Technology": 0.30,
        "Finance": 0.20,
        "Healthcare": 0.10,
        "Retail": -0.10,
        "Manufacturing": 0.05,
        "Education": -0.15,
    }

    size_effect = {
        "Small": -0.20,
        "Medium": 0.05,
        "Large": 0.25,
        "Enterprise": 0.40,
    }

    score = (
        -2.0
        + np.array([source_effect[x] for x in lead_source])
        + np.array([industry_effect[x] for x in industry])
        + np.array([size_effect[x] for x in company_size])
        + 0.075 * interactions
        + 0.12 * followups
        - 0.018 * response_time_hours
        + 0.65 * quotation_sent
        + 0.000003 * quotation_value
        + 0.025 * website_visits
        + 0.85 * previous_customer
        + 0.70 * demo_attended
        + 0.025 * salesperson_experience
        + rng.normal(0, 0.65, size=n_records)
    )

    conversion_probability = sigmoid(score)

    converted = rng.binomial(
        1,
        conversion_probability,
    )

    df = pd.DataFrame(
        {
            "lead_id": lead_id,
            "lead_source": lead_source,
            "industry": industry,
            "location": location,
            "company_size": company_size,
            "lead_age_days": lead_age_days,
            "interactions": interactions,
            "followups": followups,
            "response_time_hours": response_time_hours.round(2),
            "quotation_sent": quotation_sent,
            "quotation_value": quotation_value.round(2),
            "website_visits": website_visits,
            "previous_customer": previous_customer,
            "demo_attended": demo_attended,
            "salesperson_experience": salesperson_experience,
            "converted": converted,
        }
    )

    return df


def introduce_data_quality_issues(df, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed + 1)

    df = df.copy()

    # Small, controlled amount of missing data.
    missing_columns = [
        "lead_source",
        "industry",
        "company_size",
        "response_time_hours",
        "quotation_value",
    ]

    for column in missing_columns:
        indices = rng.choice(
            df.index,
            size=max(1, int(len(df) * 0.01)),
            replace=False,
        )
        df.loc[indices, column] = np.nan

    # Small number of invalid values.
    invalid_response_indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )
    df.loc[
        invalid_response_indices,
        "response_time_hours",
    ] = -5

    invalid_age_indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )
    df.loc[
        invalid_age_indices,
        "lead_age_days",
    ] = -10

    # Small number of duplicate records.
    duplicate_rows = df.sample(
        n=20,
        random_state=seed,
    )

    df = pd.concat(
        [df, duplicate_rows],
        ignore_index=True,
    )

    return df


def main():
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_dataset()
    df = introduce_data_quality_issues(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Dataset created: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nTarget distribution:")
    print(df["converted"].value_counts())

    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()