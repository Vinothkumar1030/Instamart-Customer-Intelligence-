
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "outputs" / "lapse_cohort.csv"

# Load cohort
df = pd.read_csv(INPUT_FILE)

# Define target and features
target = "lapsed"

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
y = df[target]

# Check missing values before preprocessing
print("Missing values:")
print(X.isnull().sum())

if X.isnull().any().any():
    raise ValueError("Missing values found in feature columns!")

# Split BEFORE fitting encoders or scalers
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data without fitting again
X_test_processed = preprocessor.transform(X_test)

print("\nPREPROCESSING VALIDATION")
print("-" * 40)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training features after encoding:", X_train_processed.shape[1])
print("Training matrix shape:", X_train_processed.shape)
print("Testing matrix shape:", X_test_processed.shape)
print("Training lapsed rate:", round(y_train.mean(), 4))
print("Testing lapsed rate:", round(y_test.mean(), 4))
print("\nPreprocessing completed successfully.")
