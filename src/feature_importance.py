import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv(
    "data/traffic_synthetic.csv"
)

data["Date"] = pd.to_datetime(
    data["Date"]
)

data["Time"] = pd.to_datetime(
    data["Time"],
    format="%H:%M"
)

data["Hour"] = data["Time"].dt.hour
data["Minute"] = data["Time"].dt.minute


# ==========================================
# 2. SORT CHRONOLOGICALLY
# ==========================================

data = data.sort_values(
    ["Date", "Hour", "Minute"]
).reset_index(drop=True)


# ==========================================
# 3. FEATURES AND TARGET
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
# 5. LOAD MODEL
# ==========================================

model = joblib.load(
    "models/traffic_model_v2.pkl"
)


# ==========================================
# 6. PERMUTATION IMPORTANCE
# ==========================================

importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="r2"
)


# ==========================================
# 7. CREATE RESULTS TABLE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance.importances_mean
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("        FEATURE IMPORTANCE")
print("==========================================")

print(
    importance_df.to_string(
        index=False
    )
)


# ==========================================
# 9. PLOT
# ==========================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel(
    "Permutation Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Traffic Prediction Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "models/feature_importance.png",
    dpi=150
)

plt.show()


print("\nFeature importance plot saved:")
print(
    "models/feature_importance.png"
)
# Save feature importance for dashboard
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": result.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("\nFeature importance saved to:")
print("models/feature_importance.csv")

# ==========================================
# SAVE FEATURE IMPORTANCE FOR DASHBOARD
# ==========================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": result.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
print("Location: models/feature_importance.csv")