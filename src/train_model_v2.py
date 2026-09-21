import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv("data/traffic_synthetic.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. DATA CLEANING
# ==========================================

data["Date"] = pd.to_datetime(data["Date"])

data["Time"] = pd.to_datetime(
    data["Time"],
    format="%H:%M"
)


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

data["Hour"] = data["Time"].dt.hour
data["Minute"] = data["Time"].dt.minute

data["DayOfWeek"] = data["Date"].dt.dayofweek


# ==========================================
# 4. FEATURES
# ==========================================

features = [
    "Hour",
    "Minute",
    "Day",
    "Weather",
    "Temperature"
]

target = "Vehicle_Count"

X = data[features]
y = data[target]


print("\nFeatures used:")
print(features)

print("\nTarget:")
print(target)


# ==========================================
# 5. CATEGORICAL / NUMERICAL FEATURES
# ==========================================

categorical_features = [
    "Day",
    "Weather"
]

numerical_features = [
    "Hour",
    "Minute",
    "Temperature"
]


# ==========================================
# 6. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 7. MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 8. PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 9. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 10. TRAIN
# ==========================================

pipeline.fit(
    X_train,
    y_train
)

print("\nImproved model training completed!")


# ==========================================
# 11. PREDICTION
# ==========================================

predictions = pipeline.predict(X_test)


# ==========================================
# 12. EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n==========================================")
print("       IMPROVED TRAFFIC MODEL")
print("==========================================")

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R2 Score:", round(r2, 4))


# ==========================================
# 13. ACTUAL VS PREDICTED
# ==========================================

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(0)
})

print("\nActual vs Predicted:")
print(comparison.to_string(index=False))


# ==========================================
# 14. SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    pipeline,
    "models/traffic_model_v2.pkl"
)

print("\n==========================================")
print("Model saved successfully!")
print("models/traffic_model_v2.pkl")
print("==========================================")