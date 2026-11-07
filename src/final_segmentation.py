
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

# Load RFM data
rfm = pd.read_csv(OUTPUT_DIR / "rfm_table.csv")

# Select features
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Final K-Means model
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["cluster"] = kmeans.fit_predict(X_scaled)

# Save customer-level segmentation
customer_output = OUTPUT_DIR / "customer_segments.csv"
rfm.to_csv(customer_output, index=False)

# Create segment profiles
profiles = rfm.groupby("cluster").agg(
    customer_count=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    median_recency=("recency", "median"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
    median_monetary=("monetary", "median")
).reset_index()

profiles["customer_percentage"] = (
    profiles["customer_count"] / len(rfm) * 100
).round(2)

profiles = profiles.round(2)

# Save segment profiles
profile_output = OUTPUT_DIR / "segment_profiles.csv"
profiles.to_csv(profile_output, index=False)

print("\nFINAL CUSTOMER SEGMENTATION")
print("-" * 50)
print("Total customers:", len(rfm))
print("Number of clusters:", rfm["cluster"].nunique())

print("\nSEGMENT PROFILES")
print(profiles.to_string(index=False))

print("\nSaved:", customer_output)
print("Saved:", profile_output)
