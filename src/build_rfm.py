
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
ORDERS_FILE = ROOT / "data" / "clean" / "orders.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Snapshot date
SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")

# Load orders
orders = pd.read_csv(ORDERS_FILE)

# Use completed orders only
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
completed = orders[orders["status"].str.lower() == "completed"].copy()

# Keep orders up to the snapshot date
completed = completed[completed["order_ts"] < SNAPSHOT_DATE]

# Calculate RFM
rfm = completed.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
).reset_index()

# Recency in whole days
rfm["recency"] = (
    SNAPSHOT_DATE - rfm["last_order_date"]
).dt.days

# Final RFM table
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].sort_values("customer_id")

# Save output
output_file = OUTPUT_DIR / "rfm_table.csv"
rfm.to_csv(output_file, index=False)

# Display validation metrics
print("RFM ANALYSIS")
print("-" * 40)
print("Customers:", len(rfm))
print("Median recency:", rfm["recency"].median())
print("Minimum recency:", rfm["recency"].min())
print("Maximum recency:", rfm["recency"].max())
print("Mean frequency:", round(rfm["frequency"].mean(), 2))
print("Maximum frequency:", rfm["frequency"].max())
print("Mean monetary:", round(rfm["monetary"].mean(), 2))
print("Maximum monetary:", round(rfm["monetary"].max(), 2))
print("Saved:", output_file)
