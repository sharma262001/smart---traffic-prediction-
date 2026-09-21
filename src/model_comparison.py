import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv("data/traffic.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

data["Date"] = pd.to_datetime(data["Date"])

data["Time"] = pd.to_datetime(
    data["Time"],
    format="%H:%M"
)

data["Hour"] = data["Time"].dt.hour
data["Minute"] = data["Time"].dt.minute
data["DayOfWeek"] = data["Date"].dt.dayofweek


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = data[
    [
        "Hour",
        "Minute",
        "DayOfWeek",
        "Temperature"
    ]
]

y = data["Vehicle_Count"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 6. TRAIN AND EVALUATE
# ==========================================

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "R2 Score": round(r2, 4)
    })


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("             MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("=" * 60)