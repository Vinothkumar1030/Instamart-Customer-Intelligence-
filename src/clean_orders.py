
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def clean_orders():
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")

    print("Original rows:", len(orders))

    # 1. Remove exact duplicate rows
    before = len(orders)
    orders = orders.drop_duplicates()
    print("Exact duplicate rows removed:", before - len(orders))

    # 2. Convert order timestamp to datetime
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    # 3. Keep only the latest record for each order_id
    orders = orders.sort_values("order_ts")
    orders = orders.drop_duplicates(
        subset="order_id", keep="last"
    )

    # 4. Convert order amount to numeric
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    # 5. Join product price and category
    product_info = products[
        ["product_id", "price_inr", "category"]
    ]

    orders = orders.merge(
        product_info,
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # 6. Fill missing order amounts using product price,
    # quantity and discount percentage
    missing_amount = orders["order_amount"].isna()

    calculated_amount = (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)

    orders.loc[missing_amount, "order_amount"] = (
        calculated_amount[missing_amount]
    )

    # 7. Convert negative delivery times to missing
    orders.loc[
        orders["delivery_minutes"] < 0,
        "delivery_minutes"
    ] = pd.NA

    # 8. Fill missing delivery time for completed orders
    completed = orders["status"].eq("completed")

    category_medians = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .median()
    )

    missing_completed = (
        completed & orders["delivery_minutes"].isna()
    )

    orders.loc[missing_completed, "delivery_minutes"] = (
        orders.loc[missing_completed, "category"]
        .map(category_medians)
    )

    # 9. Cancelled orders retain missing delivery time
    # Pending records are excluded by keeping the final
    # latest record for each order_id above.

    # Remove helper columns
    orders = orders.drop(columns=["price_inr", "category"])

    # Save cleaned data
    output_path = CLEAN_DIR / "orders_clean.csv"
    orders.to_csv(output_path, index=False)

    # 10. Validation summary
    print("\nORDERS CLEANING COMPLETED")
    print("Cleaned rows:", len(orders))
    print("Distinct order IDs:", orders["order_id"].nunique())
    print("\nStatus counts:")
    print(orders["status"].value_counts().to_string())

    print("\nMissing order amounts:", orders["order_amount"].isna().sum())

    print(
        "Missing delivery minutes in completed orders:",
        orders.loc[orders["status"].eq("completed"),
                   "delivery_minutes"].isna().sum()
    )

    print(
        "Missing delivery minutes in cancelled orders:",
        orders.loc[orders["status"].eq("cancelled"),
                   "delivery_minutes"].isna().sum()
    )

    completed_revenue = orders.loc[
        orders["status"].eq("completed"), "order_amount"
    ].sum()

    print("Completed order revenue: INR",
          f"{completed_revenue:,.2f}")

    print("Saved to:", output_path)


if __name__ == "__main__":
    clean_orders()
