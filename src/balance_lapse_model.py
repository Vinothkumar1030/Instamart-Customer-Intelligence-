
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"

df = pd.read_csv(OUTPUT / "lapse_cohort.csv")

numeric = [
    "days_since_last_order",
    "orders_last_30_days",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical = [
    "city", "gender", "favorite_category"
]

X = df[numeric + categorical]
y = df["lapsed"]

# Same reproducible train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Original training class counts
print("Original training counts:")
print(y_train.value_counts().sort_index())

# Oversample minority class using training data only
train_data = X_train.copy().reset_index(drop=True)
train_data["lapsed"] = y_train.reset_index(drop=True)

majority = train_data[train_data["lapsed"] == 0]
minority = train_data[train_data["lapsed"] == 1]

minority_oversampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_data = pd.concat(
    [majority, minority_oversampled],
    ignore_index=True
)

balanced_data = balanced_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_balanced = balanced_data.drop(columns=["lapsed"])
y_balanced = balanced_data["lapsed"]

print("\nBalanced training counts:")
print(y_balanced.value_counts().sort_index())

# Preprocessing pipeline
def make_model():
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ), categorical)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            C=10,
            max_iter=1000
        ))
    ])

# Original unbalanced model
original_model = make_model()
original_model.fit(X_train, y_train)
original_pred = original_model.predict(X_test)

# Balanced model
balanced_model = make_model()
balanced_model.fit(X_balanced, y_balanced)
balanced_pred = balanced_model.predict(X_test)

# Compare on unchanged test data
def metrics(y_true, pred):
    return {
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred)
    }

original_metrics = metrics(y_test, original_pred)
balanced_metrics = metrics(y_test, balanced_pred)

comparison = pd.DataFrame([
    {"model": "Unbalanced", **original_metrics},
    {"model": "Oversampled", **balanced_metrics}
])

comparison.to_csv(
    OUTPUT / "lapse_imbalance_comparison.csv",
    index=False
)

joblib.dump(
    balanced_model,
    OUTPUT / "lapse_model_balanced.joblib"
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nSaved comparison and balanced model.")
