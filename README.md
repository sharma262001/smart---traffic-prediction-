https://rw48ghv4vuzuhxyjr7wdvw.streamlit.app

# Smart Traffic Prediction

Smart Traffic Prediction is a machine learning project that estimates vehicle volume and congestion level using time, weather, and date-based traffic patterns. It combines data preprocessing, model training, feature analysis, and a Streamlit dashboard to make traffic insights easy to understand and use.

## Purpose

The project is designed to help cities, traffic departments, and analysts understand:

- Peak traffic hours
- Vehicle volume trends
- Congestion patterns
- Weather-related traffic behavior
- Daily traffic prediction using historical data

This helps support better traffic management, route planning, and decision-making for smart city systems.

## How it works

1. Data is loaded from the traffic dataset in the `data/` folder.
2. Date and time features are extracted, such as hour, minute, and day of week.
3. Weather and temperature are used as traffic predictors.
4. A machine learning pipeline is trained using a Random Forest Regressor.
5. The model predicts the number of vehicles for a given input.
6. The results are displayed through a Streamlit dashboard and a command-line prediction script.

The model is trained in `src/train_model_v2.py`, while the prediction interface is in `src/predict.py` and the dashboard is in `dashboard/app.py`.

## Project structure

- `dashboard/app.py` — interactive Streamlit dashboard
- `data/` — traffic datasets used for training and analysis
- `models/` — saved trained model files and evaluation charts
- `src/` — training, prediction, and analysis scripts
- `README.md` — project overview and instructions

## Requirements

Python 3.10+ is recommended.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Run the project

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train_model_v2.py
```

This creates or updates the trained model in `models/traffic_model_v2.pkl`.

### 4. Run the prediction script

```bash
python src/predict.py
```

You will be prompted to enter:

- hour
- minute
- day
- weather
- temperature

### 5. Start the dashboard

```bash
streamlit run dashboard/app.py
```

Then open the local URL shown in the terminal in your browser.

## Example output

The system predicts vehicle volume and classifies traffic as:

- LOW
- MEDIUM
- HIGH

## Notes

- The project uses synthetic or sample traffic data for demonstration and model training.
- No passwords, API keys, personal emails, or personal user data are included in this repository.
- If you later add real-world data or cloud credentials, keep them in environment variables or secret management tools instead of committing them to GitHub.

## License

This project is provided for educational and demonstration purposes.
