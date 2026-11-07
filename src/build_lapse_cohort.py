
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")

# Load datasets
customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")
products = pd.read_csv(DATA / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Completed orders only
completed = orders[
    orders["status"].str.lower() == "completed"
].copy()

# Feature data: strictly before cutoff
past_orders = completed[
    completed["order_ts"] < CUTOFF
].copy()

# Cohort: customers with at least one past completed order
cohort_ids = past_orders["customer_id"].unique()

cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

# Last completed order date
last_order = past_orders.groupby("customer_id")["order_ts"].max()

# Number of completed orders in the previous 30 days
recent_start = CUTOFF - pd.Timedelta(days=30)

recent_counts = (
    past_orders[
        past_orders["order_ts"] >= recent_start
    ]
    .groupby("customer_id")["order_id"]
    .count()
)

# Average delivery time using only pre-cutoff orders
avg_delivery = past_orders.groupby("customer_id")[
    "delivery_minutes"
].mean()

# Most frequent product category
category_orders = past_orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

def most_frequent_category(series):
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else "Unknown"

favorite_category = category_orders.groupby(
    "customer_id"
)["category"].agg(most_frequent_category)

# Build features
cohort["last_order_date"] = cohort["customer_id"].map(last_order)

cohort["days_since_last_order"] = (
    CUTOFF - cohort["last_order_date"]
).dt.days

cohort["orders_last_30_days"] = (
    cohort["customer_id"].map(recent_counts).fillna(0)
)

cohort["avg_delivery_minutes"] = (
    cohort["customer_id"].map(avg_delivery)
)

cohort["favorite_category"] = (
    cohort["customer_id"].map(favorite_category)
)

cohort["tenure_days"] = (
    CUTOFF - cohort["signup_date"]
).dt.days

# Label using September completed orders
september_orders = completed[
    (completed["order_ts"] >= CUTOFF) &
    (completed["order_ts"] < LABEL_END)
]

active_ids = set(september_orders["customer_id"])

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(active_ids)
).astype(int)

# Final output
columns = [
    "customer_id",
    "days_since_last_order",
    "orders_last_30_days",
    "avg_delivery_minutes",
    "favorite_category",
    "city",
    "age",
    "gender",
    "tenure_days",
    "lapsed"
]

cohort = cohort[columns].sort_values("customer_id")

output_file = OUTPUT / "lapse_cohort.csv"
cohort.to_csv(output_file, index=False)

print("\nLAPSE COHORT SUMMARY")
print("-" * 40)
print("Cohort customers:", len(cohort))
print("Lapsed customers:", int(cohort["lapsed"].sum()))
print("Active customers:", int((cohort["lapsed"] == 0).sum()))
print(
    "Lapsed percentage:",
    round(cohort["lapsed"].mean() * 100, 2)
)

print("\nSaved:", output_file)
