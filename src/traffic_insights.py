import pandas as pd


# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("data/traffic.csv")

print("Traffic dataset loaded successfully!")


# ==========================================
# DATA PROCESSING
# ==========================================

data["Time"] = pd.to_datetime(
    data["Time"],
    format="%H:%M"
)

data["Hour"] = data["Time"].dt.hour


# ==========================================
# BASIC STATISTICS
# ==========================================

print("\n==========================================")
print("        SMART TRAFFIC INSIGHTS")
print("==========================================")


print("\nTotal Records:")
print(len(data))


print("\nTotal Vehicles:")
print(data["Vehicle_Count"].sum())


print("\nAverage Vehicle Count:")
print(round(data["Vehicle_Count"].mean(), 2))


print("\nAverage Speed:")
print(
    round(data["Average_Speed"].mean(), 2),
    "km/h"
)


# ==========================================
# PEAK TRAFFIC HOUR
# ==========================================

hourly_traffic = (
    data.groupby("Hour")["Vehicle_Count"]
    .mean()
)


peak_hour = hourly_traffic.idxmax()
peak_vehicle_count = hourly_traffic.max()


print("\n------------------------------------------")
print("PEAK TRAFFIC")
print("------------------------------------------")

print(
    f"Peak Hour: {peak_hour}:00"
)

print(
    f"Average Vehicles: {peak_vehicle_count:.0f}"
)


# ==========================================
# LOWEST TRAFFIC HOUR
# ==========================================

lowest_hour = hourly_traffic.idxmin()
lowest_vehicle_count = hourly_traffic.min()


print("\n------------------------------------------")
print("LOWEST TRAFFIC")
print("------------------------------------------")

print(
    f"Lowest Traffic Hour: {lowest_hour}:00"
)

print(
    f"Average Vehicles: {lowest_vehicle_count:.0f}"
)


# ==========================================
# MAXIMUM TRAFFIC RECORD
# ==========================================

max_row = data.loc[
    data["Vehicle_Count"].idxmax()
]


print("\n------------------------------------------")
print("HIGHEST TRAFFIC RECORD")
print("------------------------------------------")

print(
    "Date:",
    max_row["Date"]
)

print(
    "Time:",
    max_row["Time"].strftime("%H:%M")
)

print(
    "Vehicle Count:",
    max_row["Vehicle_Count"]
)

print(
    "Weather:",
    max_row["Weather"]
)

print(
    "Congestion:",
    max_row["Congestion"]
)


# ==========================================
# CONGESTION ANALYSIS
# ==========================================

most_common_congestion = (
    data["Congestion"]
    .value_counts()
    .idxmax()
)


print("\n------------------------------------------")
print("CONGESTION ANALYSIS")
print("------------------------------------------")

print(
    "Most Common Congestion Level:",
    most_common_congestion
)


# ==========================================
# WEATHER ANALYSIS
# ==========================================

weather_traffic = (
    data.groupby("Weather")["Vehicle_Count"]
    .mean()
    .sort_values(ascending=False)
)


print("\n------------------------------------------")
print("WEATHER ANALYSIS")
print("------------------------------------------")

print(weather_traffic)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n==========================================")
print("          TRAFFIC SUMMARY")
print("==========================================")

print(
    f"Peak Hour: {peak_hour}:00"
)

print(
    f"Peak Vehicles: {peak_vehicle_count:.0f}"
)

print(
    f"Lowest Hour: {lowest_hour}:00"
)

print(
    f"Average Speed: {data['Average_Speed'].mean():.2f} km/h"
)

print(
    f"Common Congestion: {most_common_congestion}"
)

print("==========================================")
