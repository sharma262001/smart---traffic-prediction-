import joblib
import pandas as pd


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/traffic_model_v2.pkl")


print("==========================================")
print("       SMART TRAFFIC PREDICTION")
print("==========================================")


# ==========================================
# USER INPUT
# ==========================================

hour = int(input("Enter hour (0-23): "))
minute = int(input("Enter minute (0-59): "))

day = input(
    "Enter day (Monday-Sunday): "
).strip().capitalize()

weather = input(
    "Enter weather (Clear/Rain/Cloudy/Fog): "
).strip().capitalize()

temperature = float(
    input("Enter temperature (°C): ")
)


# ==========================================
# VALIDATE DAY
# ==========================================

valid_days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

if day not in valid_days:
    print("\nInvalid day entered.")
    exit()


# ==========================================
# CREATE INPUT
# ==========================================

input_data = pd.DataFrame({
    "Hour": [hour],
    "Minute": [minute],
    "Day": [day],
    "Weather": [weather],
    "Temperature": [temperature]
})


# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(input_data)[0]

prediction = max(
    0,
    round(prediction)
)


# ==========================================
# TRAFFIC LEVEL
# ==========================================

if prediction < 800:

    traffic_level = "LOW"

elif prediction < 1400:

    traffic_level = "MEDIUM"

else:

    traffic_level = "HIGH"


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("          PREDICTION RESULT")
print("==========================================")

print(
    f"Predicted Vehicle Count: {prediction:,}"
)

print(
    f"Traffic Level: {traffic_level}"
)

print("==========================================")