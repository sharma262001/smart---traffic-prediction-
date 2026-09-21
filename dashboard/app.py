import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

data = pd.read_csv("data/traffic_synthetic.csv")
model = joblib.load("models/traffic_model_v2.pkl")

# ==========================================
# TRAFFIC INSIGHTS
# ==========================================

# ==========================================
# TIME PROCESSING
# ==========================================

data["Time"] = pd.to_datetime(
    data["Time"].astype(str),
    format="%H:%M",
    errors="coerce"
)

data["Time"] = pd.to_datetime(data["Time"])
data["Hour"] = data["Time"].dt.hour
hourly_traffic = (
    data.groupby("Hour")["Vehicle_Count"]
    .mean()
)

peak_hour = hourly_traffic.idxmax()

peak_vehicles = hourly_traffic.max()

lowest_hour = hourly_traffic.idxmin()

lowest_vehicles = hourly_traffic.min()

most_common_congestion = (
    data["Congestion"]
    .value_counts()
    .idxmax()
)
# ==========================================
# DATA PROCESSING
# ==========================================

data["Date"] = pd.to_datetime(data["Date"])

data["Time"] = pd.to_datetime(
    data["Time"],
    format="%H:%M"
)

data["Hour"] = data["Time"].dt.hour


# ==========================================
# HEADER
# ==========================================

st.title("🚦 Smart Traffic Prediction System")

st.markdown(
    "### AI-powered traffic analysis and vehicle prediction"
)

st.divider()


# ==========================================
# KEY METRICS
# ==========================================

st.subheader("📊 Traffic Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        len(data)
    )

with col2:
    st.metric(
        "Total Vehicles",
        f"{data['Vehicle_Count'].sum():,}"
    )

with col3:
    st.metric(
        "Average Vehicles",
        f"{data['Vehicle_Count'].mean():,.0f}"
    )

with col4:
    st.metric(
        "Average Speed",
        f"{data['Average_Speed'].mean():.1f} km/h"
    )


st.divider()


# ==========================================
# TRAFFIC TREND
# ==========================================

st.subheader("📈 Vehicle Count Trend")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    data.index,
    data["Vehicle_Count"],
    marker="o"
)

ax.set_xlabel("Record")
ax.set_ylabel("Vehicle Count")
ax.set_title("Traffic Volume Over Records")

st.pyplot(fig)


# ==========================================
# TRAFFIC BY HOUR
# ==========================================

st.subheader("🕐 Traffic by Hour")

hourly_traffic = (
    data.groupby("Hour")["Vehicle_Count"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    hourly_traffic["Hour"],
    hourly_traffic["Vehicle_Count"],
    marker="o"
)

ax.set_xlabel("Hour")
ax.set_ylabel("Average Vehicle Count")
ax.set_title("Average Traffic by Hour")

st.pyplot(fig)


# ==========================================
# CONGESTION ANALYSIS
# ==========================================

st.subheader("🚦 Congestion Analysis")

col1, col2 = st.columns(2)

with col1:

    congestion_counts = data["Congestion"].value_counts()

    st.bar_chart(congestion_counts)


with col2:

    st.dataframe(
        data[
            [
                "Time",
                "Vehicle_Count",
                "Average_Speed",
                "Congestion"
            ]
        ],
        use_container_width=True
    )


# ==========================================
# WEATHER ANALYSIS
# ==========================================

st.subheader("🌤️ Traffic by Weather")

weather_traffic = (
    data.groupby("Weather")["Vehicle_Count"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(weather_traffic)


# ==========================================
# SPEED VS TRAFFIC
# ==========================================

st.subheader("🚗 Vehicle Count vs Average Speed")

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(
    data["Average_Speed"],
    data["Vehicle_Count"]
)

ax.set_xlabel("Average Speed (km/h)")
ax.set_ylabel("Vehicle Count")
ax.set_title("Relationship Between Speed and Traffic Volume")

st.pyplot(fig)


# ==========================================
# PREDICTION SECTION
# ==========================================

# ==========================================
# PREDICTION SECTION
# ==========================================

st.divider()

st.header("🔮 Traffic Prediction")

st.write(
    "Enter traffic conditions to predict the expected vehicle count."
)


col1, col2 = st.columns(2)

with col1:

    hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=18
    )

    minute = st.slider(
        "Minute",
        min_value=0,
        max_value=59,
        value=0
    )

    day = st.selectbox(
        "Day",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


with col2:

    weather = st.selectbox(
        "Weather",
        [
            "Clear",
            "Rain",
            "Cloudy",
            "Fog"
        ]
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=22.0
    )

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
## ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.subheader("📊 Feature Importance")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMPORTANCE_FILE = PROJECT_ROOT / "models" / "feature_importance.csv"

if IMPORTANCE_FILE.exists():

    importance_data = pd.read_csv(IMPORTANCE_FILE)

    importance_data = importance_data.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        importance_data.set_index("Feature")
    )

    st.dataframe(
        importance_data,
        hide_index=True,
        width="stretch"
    )

else:

    st.error(
        f"Feature importance file not found at: {IMPORTANCE_FILE}"
    )
# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(
    "🔮 Predict Traffic",
    width="stretch"
):

    input_data = pd.DataFrame({
        "Hour": [hour],
        "Minute": [minute],
        "Day": [day],
        "Weather": [weather],
        "Temperature": [temperature]
    })

    prediction = model.predict(input_data)[0]

    prediction = max(
        0,
        round(prediction)
    )


    # ======================================
    # TRAFFIC LEVEL
    # ======================================

    if prediction < 800:

        traffic_level = "LOW"

    elif prediction < 1400:

        traffic_level = "MEDIUM"

    else:

        traffic_level = "HIGH"


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.divider()

    result1, result2 = st.columns(2)

    with result1:

        st.metric(
            "Predicted Vehicles",
            f"{prediction:,}"
        )

    with result2:

        st.metric(
            "Traffic Level",
            traffic_level
        )


    if traffic_level == "LOW":

        st.success(
            "🟢 Traffic is expected to be LOW."
        )

    elif traffic_level == "MEDIUM":

        st.warning(
            "🟡 Traffic is expected to be MEDIUM."
        )

    else:

        st.error(
            "🔴 Traffic is expected to be HIGH."
        )