
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]

# Load RFM data
rfm = pd.read_csv(ROOT / "outputs" / "rfm_table.csv")

# Select RFM features
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test K values from 2 to 8
results = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette
    })

# Save results
metrics = pd.DataFrame(results)

output_file = ROOT / "outputs" / "kmeans_metrics.csv"
metrics.to_csv(output_file, index=False)

print("\nK-MEANS EVALUATION")
print("-" * 40)
print(metrics.to_string(index=False))

print("\nSaved:", output_file)
