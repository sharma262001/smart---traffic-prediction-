import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv("data/traffic.csv")

print("Dataset loaded successfully!")
print(data.head())


# ==========================================
# 2. CONVERT TIME
# ==========================================

data["Time"] = pd.to_datetime(
    data["Time"],
    format="%H:%M"
)

data["Hour"] = data["Time"].dt.hour
data["Minute"] = data["Time"].dt.minute


# ==========================================
# 3. CONVERT DATE
# ==========================================

data["Date"] = pd.to_datetime(data["Date"])

data["DayOfWeek"] = data["Date"].dt.dayofweek


# ==========================================
# 4. FEATURES AND TARGET
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
# 5. TRAIN / TEST SPLIT
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
# 6. CREATE MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# ==========================================
# 8. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== TRAFFIC PREDICTION RESULTS ==========")

print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 4))


# ==========================================
# 10. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/traffic_model.pkl")

print("\nModel saved successfully!")
print("Location: models/traffic_model.pkl")


# ==========================================
# 11. ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values,
    marker="o",
    label="Actual Traffic"
)

plt.plot(
    y_pred,
    marker="o",
    label="Predicted Traffic"
)

plt.title("Actual vs Predicted Traffic")
plt.xlabel("Test Sample")
plt.ylabel("Vehicle Count")

plt.legend()

plt.tight_layout()
plt.show()