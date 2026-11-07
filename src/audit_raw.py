
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

FILES = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "tickets.csv",
]

def audit_file(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    print("\n" + "=" * 65)
    print(f"FILE: {filename}")
    print("=" * 65)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")

    print("\nDATA TYPES:")
    print(df.dtypes.to_string())

    print("\nMISSING VALUES:")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (df.isna().mean() * 100).round(2)
    })
    print(missing.to_string())

    print("\nBLANK STRING COUNTS:")
    blanks = df.select_dtypes(include="object").apply(
        lambda col: col.astype("string").str.strip().eq("").sum()
    )
    print(blanks.to_string())


def audit_specific_columns():
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    tickets = pd.read_csv(DATA_DIR / "tickets.csv")

    print("\n" + "=" * 65)
    print("PROJECT-SPECIFIC AUDIT")
    print("=" * 65)

    # Orders
    print("\nORDERS:")
    if "order_id" in orders.columns:
        print("Distinct order IDs:", orders["order_id"].nunique())
        print("Duplicate order ID rows:",
              orders["order_id"].duplicated().sum())

    if "status" in orders.columns:
        print("\nOrder status counts:")
        print(orders["status"].value_counts(dropna=False).to_string())

    for col in ["delivery_minutes", "discount_pct", "order_amount"]:
        if col in orders.columns:
            print(f"Blank/missing {col}:",
                  orders[col].isna().sum()
                  + orders[col].astype("string").str.strip().eq("").sum())

    # Customers
    print("\nCUSTOMERS:")
    if "age" in customers.columns:
        age = pd.to_numeric(customers["age"], errors="coerce")
        print("Age values below 10:", (age < 10).sum())
        print("Age values above 100:", (age > 100).sum())
        print("Age values outside 10-100:",
              ((age < 10) | (age > 100)).sum())

    for col in ["city", "gender"]:
        if col in customers.columns:
            print(f"\n{col} value counts:")
            print(customers[col].value_counts(dropna=False).to_string())

    # Tickets
    print("\nTICKETS:")
    if "ticket_text" in tickets.columns:
        blank_tickets = (
            tickets["ticket_text"].isna()
            | tickets["ticket_text"].astype("string").str.strip().eq("")
        )
        print("Blank ticket text rows:", blank_tickets.sum())


if __name__ == "__main__":
    print("INSTAMART RAW DATA AUDIT")

    for filename in FILES:
        audit_file(filename)

    audit_specific_columns()

    print("\nRAW DATA AUDIT COMPLETED")
