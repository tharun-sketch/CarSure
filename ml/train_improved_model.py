from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------
# CarSure Improved ML Model - Step 8
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "carsure_ml_development_dataset.csv"
MODEL_PATH = BASE_DIR / "ml" / "carsure_price_model.pkl"

CURRENT_YEAR = 2026
RANDOM_STATE = 42


# Load development dataset
df = pd.read_csv(DATASET_PATH)

# Create vehicle age from manufacturing year
df["vehicle_age"] = CURRENT_YEAR - df["manufacture_year"]

# Features used by the valuation model
features = [
    "brand",
    "model",
    "variant",
    "fuel_type",
    "transmission",
    "manufacture_year",
    "vehicle_age",
    "current_km",
    "owner_count",
    "registration_location",
]

target = "current_market_value"

X = df[features]
y = df[target]

categorical_features = [
    "brand",
    "model",
    "variant",
    "fuel_type",
    "transmission",
    "registration_location",
]

numeric_features = [
    "manufacture_year",
    "vehicle_age",
    "current_km",
    "owner_count",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numeric", "passthrough", numeric_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    max_depth=12,
    min_samples_leaf=2,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# Proper train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print("=" * 60)
print("CarSure - Improved Used Car Valuation Model")
print("=" * 60)

print(f"Dataset records : {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")

# Train
pipeline.fit(X_train, y_train)

# Test
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("-" * 60)
print(f"MAE : ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R²  : {r2:.4f}")

# Save improved model
joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)

print("\nImportant:")
print("This model was trained on development/sample data.")
print("Evaluation results describe this development dataset and")
print("should not be treated as real-world market accuracy.")
print("=" * 60)
