
from pathlib import Path
import sqlite3
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
DB_PATH = DATA_DIR / "instamart.db"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def store_and_validate():
    # Load the cleaned tables
    customers = pd.read_csv(CLEAN_DIR / "customers_clean.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders_clean.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")

    # Store final CSV files with required names
    customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
    products.to_csv(CLEAN_DIR / "products.csv", index=False)
    orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

    # Store all three tables in SQLite
    with sqlite3.connect(DB_PATH) as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)

        # Verify database table row counts
        db_counts = {
            table: pd.read_sql_query(
                f"SELECT COUNT(*) AS n FROM {table}", conn
            )["n"].iloc[0]
            for table in ["customers", "products", "orders"]
        }

    # Validation checkpoints
    checks = {
        "Customers: 1,200 rows": len(customers) == 1200,
        "Products: 40 rows": len(products) == 40,
        "Orders: 12,309 rows": len(orders) == 12309,
        "Customers: no missing values": customers.isna().sum().sum() == 0,
        "Customers: valid age range": customers["age"].between(10, 100).all(),
        "Customers: standardized gender": set(customers["gender"].unique()).issubset(
            {"Male", "Female", "Unknown"}
        ),
        "Orders: unique order IDs": orders["order_id"].nunique() == 12309,
        "Orders: no pending records": not orders["status"].eq("pending").any(),
        "Orders: no missing amounts": orders["order_amount"].isna().sum() == 0,
        "Orders: completed delivery times present": orders.loc[
            orders["status"].eq("completed"), "delivery_minutes"
        ].notna().all(),
        "Orders: cancelled delivery times missing": orders.loc[
            orders["status"].eq("cancelled"), "delivery_minutes"
        ].isna().all(),
        "Orders: completed revenue matches reference": abs(
            orders.loc[orders["status"].eq("completed"), "order_amount"].sum()
            - 2998668.80
        ) < 0.01,
        "SQLite: customers row count": db_counts["customers"] == 1200,
        "SQLite: products row count": db_counts["products"] == 40,
        "SQLite: orders row count": db_counts["orders"] == 12309,
    }

    print("\nVALIDATION CHECKPOINTS")
    print("-" * 50)

    for description, passed in checks.items():
        print(f"{description}: {'PASS' if passed else 'FAIL'}")

    if all(checks.values()):
        print("\nALL CHECKPOINTS PASSED")
    else:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Validation failed: {failed}")

    print("\nCSV files saved in:", CLEAN_DIR)
    print("SQLite database saved at:", DB_PATH)


if __name__ == "__main__":
    store_and_validate()
