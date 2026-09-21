import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv("data/traffic_synthetic.csv")

data["Date"] = pd.to_datetime(data["Date"])

data["Time"] = pd.to_datetime(
    data["Time"],
    format="%H:%M"
)

data["Hour"] = data["Time"].dt.hour
data["Minute"] = data["Time"].dt.minute


# ==========================================
# 2. SORT DATA
# ==========================================

data = data.sort_values(
    ["Date", "Hour"]
).reset_index(drop=True)


# ==========================================
# 3. FEATURES
# ==========================================

features = [
    "Hour",
    "Minute",
    "Day",
    "Weather",
    "Temperature"
]

X = data[features]
y = data["Vehicle_Count"]


# ==========================================
# 4. TIME-BASED TEST DATA
# ==========================================

split_index = int(len(data) * 0.80)

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "models/traffic_model_v2.pkl"
)


# ==========================================
# 6. PREDICTIONS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 7. METRICS
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("==========================================")
print("       MODEL EVALUATION")
print("==========================================")

print(
    f"MAE  : {mae:.2f}"
)

print(
    f"MSE  : {mse:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"R2   : {r2:.4f}"
)


# ==========================================
# 8. ACTUAL VS PREDICTED DATAFRAME
# ==========================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

results["Error"] = (
    results["Actual"]
    - results["Predicted"]
)

print("\nFirst 15 predictions:")

print(
    results.head(15).round(2).to_string(
        index=False
    )
)


# ==========================================
# 9. ACTUAL VS PREDICTED PLOT
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    results["Actual"].values,
    label="Actual"
)

plt.plot(
    results["Predicted"].values,
    label="Predicted"
)

plt.title(
    "Actual vs Predicted Vehicle Count"
)

plt.xlabel(
    "Test Observation"
)

plt.ylabel(
    "Vehicle Count"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "models/actual_vs_predicted.png",
    dpi=150
)

plt.show()


# ==========================================
# 10. ERROR DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(
    results["Error"],
    bins=30
)

plt.title(
    "Prediction Error Distribution"
)

plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    "models/error_distribution.png",
    dpi=150
)

plt.show()


print("\nEvaluation plots saved:")
print("models/actual_vs_predicted.png")
print("models/error_distribution.png")