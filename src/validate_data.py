from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data" / "raw" / "leads_raw.csv"
VALIDATED_FILE = BASE_DIR / "data" / "processed" / "leads_validated.csv"


def validate_dataset(df):
    print("=" * 70)
    print("LEAD CONVERSION DATA QUALITY VALIDATION")
    print("=" * 70)

    print("\n1. DATASET SHAPE")
    print("-" * 70)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\n2. COLUMN NAMES")
    print("-" * 70)
    print(df.columns.tolist())

    print("\n3. DATA TYPES")
    print("-" * 70)
    print(df.dtypes)

    print("\n4. MISSING VALUES")
    print("-" * 70)

    missing = df.isna().sum()
    missing_percentage = (missing / len(df) * 100).round(2)

    missing_report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percentage": missing_percentage,
        }
    )

    print(missing_report[missing_report["missing_count"] > 0])

    print("\n5. DUPLICATE RECORDS")
    print("-" * 70)

    duplicate_count = df.duplicated().sum()

    print(f"Exact duplicate rows: {duplicate_count}")

    print("\n6. TARGET DISTRIBUTION")
    print("-" * 70)

    print(df["converted"].value_counts())
    print("\nTarget proportions:")
    print(df["converted"].value_counts(normalize=True).round(4))

    print("\n7. UNIQUE VALUES")
    print("-" * 70)

    categorical_columns = [
        "lead_source",
        "industry",
        "location",
        "company_size",
    ]

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))

    print("\n8. NUMERICAL RANGE CHECKS")
    print("-" * 70)

    numerical_columns = [
        "lead_age_days",
        "interactions",
        "followups",
        "response_time_hours",
        "quotation_value",
        "website_visits",
        "salesperson_experience",
    ]

    for column in numerical_columns:
        print(f"\n{column}:")
        print(f"Minimum : {df[column].min()}")
        print(f"Maximum : {df[column].max()}")

    print("\n9. INVALID VALUES")
    print("-" * 70)

    invalid_lead_age = (df["lead_age_days"] < 0).sum()
    invalid_interactions = (df["interactions"] < 0).sum()
    invalid_followups = (df["followups"] < 0).sum()
    invalid_response_time = (df["response_time_hours"] < 0).sum()
    invalid_quotation_value = (df["quotation_value"] < 0).sum()
    invalid_website_visits = (df["website_visits"] < 0).sum()
    invalid_salesperson_experience = (
        (df["salesperson_experience"] < 1)
        | (df["salesperson_experience"] > 50)
    ).sum()

    invalid_report = {
        "lead_age_days": invalid_lead_age,
        "interactions": invalid_interactions,
        "followups": invalid_followups,
        "response_time_hours": invalid_response_time,
        "quotation_value": invalid_quotation_value,
        "website_visits": invalid_website_visits,
        "salesperson_experience": invalid_salesperson_experience,
    }

    for column, count in invalid_report.items():
        print(f"{column}: {count}")

    print("\n10. QUOTATION CONSISTENCY")
    print("-" * 70)

    quotation_without_value = (
        (df["quotation_sent"] == 0)
        & (df["quotation_value"].fillna(0) > 0)
    ).sum()

    quotation_sent_without_value = (
        (df["quotation_sent"] == 1)
        & (df["quotation_value"].isna())
    ).sum()

    print(
        "Quotation value present when quotation_sent = 0:",
        quotation_without_value,
    )

    print(
        "Quotation value missing when quotation_sent = 1:",
        quotation_sent_without_value,
    )


def clean_dataset(df):
    print("\n" + "=" * 70)
    print("DATA QUALITY CLEANING")
    print("=" * 70)

    cleaned = df.copy()

    original_rows = len(cleaned)

    # ---------------------------------------------------------
    # Remove exact duplicate records
    # ---------------------------------------------------------

    duplicate_count = cleaned.duplicated().sum()

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    print(f"\nExact duplicates removed: {duplicate_count}")

    # ---------------------------------------------------------
    # Correct impossible numerical values
    #
    # Invalid values are converted to NaN.
    # We DO NOT impute them here.
    #
    # Imputation will later be learned from the training data
    # inside the sklearn preprocessing pipeline.
    # ---------------------------------------------------------

    invalid_age = cleaned["lead_age_days"] < 0
    invalid_response = cleaned["response_time_hours"] < 0

    age_count = invalid_age.sum()
    response_count = invalid_response.sum()

    cleaned.loc[invalid_age, "lead_age_days"] = pd.NA
    cleaned.loc[invalid_response, "response_time_hours"] = pd.NA

    print(f"Invalid lead_age_days corrected to missing: {age_count}")
    print(
        "Invalid response_time_hours corrected to missing:",
        response_count,
    )

    # ---------------------------------------------------------
    # Save validated dataset
    # ---------------------------------------------------------

    VALIDATED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        VALIDATED_FILE,
        index=False,
    )

    print("\nVALIDATION SUMMARY")
    print("-" * 70)
    print(f"Original rows : {original_rows}")
    print(f"Final rows    : {len(cleaned)}")
    print(f"Rows removed  : {original_rows - len(cleaned)}")

    print(f"\nValidated dataset saved to:")
    print(VALIDATED_FILE)

    return cleaned


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    validate_dataset(df)

    cleaned_df = clean_dataset(df)

    print("\nFinal validated dataset shape:")
    print(cleaned_df.shape)

    print("\nRemaining missing values:")
    print(cleaned_df.isna().sum())


if __name__ == "__main__":
    main()