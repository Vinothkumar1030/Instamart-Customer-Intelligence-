
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def clean_customers():
    input_path = RAW_DIR / "customers.csv"
    output_path = CLEAN_DIR / "customers_clean.csv"

    df = pd.read_csv(input_path)

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print("Exact duplicate rows removed:", before - len(df))

    # Standardize city names
    df["city"] = (
        df["city"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # Standardize gender values
    gender_map = {
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female",
    }

    df["gender"] = (
        df["gender"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(gender_map)
    )

    # Convert age to numeric
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Treat ages outside 10-100 as invalid
    invalid_age = ~df["age"].between(10, 100)
    print("Invalid ages:", invalid_age.sum())
    df.loc[invalid_age, "age"] = pd.NA

    # Fill missing age using median
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)

    # Keep missing gender as Unknown
    df["gender"] = df["gender"].fillna("Unknown")

    # Save cleaned customer data
    df.to_csv(output_path, index=False)

    print("\nCUSTOMER CLEANING COMPLETED")
    print("Rows:", len(df))
    print("Missing values:\n", df.isna().sum())
    print("Saved to:", output_path)


if __name__ == "__main__":
    clean_customers()
