import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ==========================================
# SETTINGS
# ==========================================

np.random.seed(42)

start_date = datetime(2026, 1, 1)

number_of_days = 90

records = []


# ==========================================
# GENERATE TRAFFIC DATA
# ==========================================

for day_number in range(number_of_days):

    current_date = start_date + timedelta(
        days=day_number
    )

    day_name = current_date.strftime("%A")

    for hour in range(24):

        # ----------------------------------
        # BASE TRAFFIC
        # ----------------------------------

        if 7 <= hour <= 9:

            base_vehicles = 1400

        elif 17 <= hour <= 20:

            base_vehicles = 1800

        elif 10 <= hour <= 16:

            base_vehicles = 1000

        elif 21 <= hour <= 23:

            base_vehicles = 700

        else:

            base_vehicles = 400


        # ----------------------------------
        # WEEKEND EFFECT
        # ----------------------------------

        if day_name in ["Saturday", "Sunday"]:

            base_vehicles *= 0.75


        # ----------------------------------
        # WEATHER
        # ----------------------------------

        weather = np.random.choice(
            [
                "Clear",
                "Cloudy",
                "Rain",
                "Fog"
            ],
            p=[
                0.60,
                0.20,
                0.15,
                0.05
            ]
        )


        # ----------------------------------
        # TEMPERATURE
        # ----------------------------------

        temperature = round(
            18
            + 10 * np.sin(
                (hour - 6) / 24 * 2 * np.pi
            )
            + np.random.normal(0, 2),
            1
        )


        # ----------------------------------
        # WEATHER EFFECT
        # ----------------------------------

        weather_factor = {
            "Clear": 1.00,
            "Cloudy": 0.95,
            "Rain": 0.75,
            "Fog": 0.80
        }

        base_vehicles *= weather_factor[weather]


        # ----------------------------------
        # RANDOM TRAFFIC VARIATION
        # ----------------------------------

        vehicle_count = int(
            max(
                100,
                base_vehicles
                + np.random.normal(0, 150)
            )
        )


        # ----------------------------------
        # AVERAGE SPEED
        # ----------------------------------

        average_speed = (
            60
            - vehicle_count / 100
            + np.random.normal(0, 3)
        )

        average_speed = round(
            max(15, min(65, average_speed)),
            1
        )


        # ----------------------------------
        # CONGESTION
        # ----------------------------------

        if vehicle_count < 800:

            congestion = "Low"

        elif vehicle_count < 1500:

            congestion = "Medium"

        else:

            congestion = "High"


        # ----------------------------------
        # SAVE RECORD
        # ----------------------------------

        records.append({
            "Date": current_date.strftime("%Y-%m-%d"),
            "Time": f"{hour:02d}:00",
            "Day": day_name,
            "Weather": weather,
            "Temperature": temperature,
            "Vehicle_Count": vehicle_count,
            "Average_Speed": average_speed,
            "Congestion": congestion
        })


# ==========================================
# CREATE DATAFRAME
# ==========================================

data = pd.DataFrame(records)


# ==========================================
# SAVE DATASET
# ==========================================

data.to_csv(
    "data/traffic_synthetic.csv",
    index=False
)


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print("==========================================")
print("   SYNTHETIC TRAFFIC DATASET CREATED")
print("==========================================")

print(
    "Rows:",
    len(data)
)

print(
    "Columns:",
    len(data.columns)
)

print("\nDataset preview:")

print(data.head())

print("\nSaved to:")

print(
    "data/traffic_synthetic.csv"
)

print("==========================================")