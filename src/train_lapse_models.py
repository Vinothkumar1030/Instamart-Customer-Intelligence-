
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

# Load cohort
df = pd.read_csv(OUTPUT / "lapse_cohort.csv")

numeric_features = [
    "days_since_last_order",
    "orders_last_30_days",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favorite_category"
]

X = df[numeric_features + categorical_features]
y = df["lapsed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Preprocessing inside pipeline prevents leakage during CV
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ), categorical_features)
])

# Baseline model
baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DummyClassifier(strategy="most_frequent"))
])

baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        )
    }

results = []

baseline_metrics = get_metrics(y_test, baseline_pred)
results.append({"model": "Baseline", **baseline_metrics})

print("BASELINE RESULTS")
print(baseline_metrics)

# Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_grid = GridSearchCV(
    lr_pipeline,
    {"model__C": [0.1, 1, 10]},
    cv=5,
    scoring="f1",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)
lr_pred = lr_grid.predict(X_test)
lr_metrics = get_metrics(y_test, lr_pred)

results.append({
    "model": "LogisticRegression",
    **lr_metrics
})

print("\nLOGISTIC REGRESSION")
print("Best parameters:", lr_grid.best_params_)
print("Best CV F1:", round(lr_grid.best_score_, 4))
print("Test metrics:", lr_metrics)
print("Confusion matrix:")
print(confusion_matrix(y_test, lr_pred))

# Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
rf_pred = rf_grid.predict(X_test)
rf_metrics = get_metrics(y_test, rf_pred)

results.append({
    "model": "RandomForest",
    **rf_metrics
})

print("\nRANDOM FOREST")
print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", round(rf_grid.best_score_, 4))
print("Test metrics:", rf_metrics)
print("Confusion matrix:")
print(confusion_matrix(y_test, rf_pred))

# Select model using cross-validation F1
if rf_grid.best_score_ >= lr_grid.best_score_:
    best_model = rf_grid.best_estimator_
    best_name = "RandomForest"
    best_pred = rf_pred
else:
    best_model = lr_grid.best_estimator_
    best_name = "LogisticRegression"
    best_pred = lr_pred

# Save selected model
model_path = OUTPUT / "lapse_model.joblib"
joblib.dump(best_model, model_path)

# Save metrics and test predictions
pd.DataFrame(results).to_csv(
    OUTPUT / "lapse_model_metrics.csv",
    index=False
)

predictions = pd.DataFrame({
    "actual": y_test,
    "predicted": best_pred
})
predictions.to_csv(
    OUTPUT / "lapse_test_predictions.csv",
    index=False
)

print("\nSELECTED MODEL:", best_name)
print("Saved model:", model_path)
print("Saved metrics:", OUTPUT / "lapse_model_metrics.csv")
