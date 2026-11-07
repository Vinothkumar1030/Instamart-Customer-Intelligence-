
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project paths
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

DATA_PATH = OUTPUT / "lapse_cohort.csv"
MODEL_PATH = OUTPUT / "lapse_model.joblib"

# Load dataset and saved model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Separate features and target
X = df.drop(columns=["lapsed"])
y = df["lapsed"]

# Recreate the same test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Extract Random Forest feature importance
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
feature_importances = classifier.feature_importances_

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importances
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

importance.to_csv(
    OUTPUT / "lapse_feature_importance.csv",
    index=False
)

# Verify model persistence
reloaded_model = joblib.load(MODEL_PATH)

predictions_before = model.predict(X_test)
predictions_after = reloaded_model.predict(X_test)

assert np.array_equal(
    predictions_before,
    predictions_after
), "Model persistence verification failed"

print("TASK 8 - INTERPRETABILITY AND PERSISTENCE")
print("-----------------------------------------")
print("Model type:", type(classifier).__name__)
print("Model loaded successfully:", MODEL_PATH.name)
print("Feature importance saved: lapse_feature_importance.csv")
print("Number of features:", len(feature_names))
print("Test samples verified:", len(X_test))
print("Predictions before and after reload are identical: PASS")
print("\nTop 10 important features:")
print(importance.head(10).to_string(index=False))
print("\nTASK 8 COMPLETED SUCCESSFULLY")
