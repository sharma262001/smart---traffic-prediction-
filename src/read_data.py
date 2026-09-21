import pandas as pd

# Load the traffic dataset
data = pd.read_csv("data/traffic.csv")

# Display the first 5 rows
print(data.head())

# Display number of rows and columns
print("\nDataset shape:")
print(data.shape)

# Display column names
print("\nColumns:")
print(data.columns)

print("\nBasic Statistics:")
print(data.describe())

print("\nHighest Vehicle Count:")
print(data["Vehicle_Count"].max())

print("\nLowest Average Speed:")
print(data["Average_Speed"].min())

print("\nCongestion Distribution:")
print(data["Congestion"].value_counts())

print("\nBasic Statistics:")
print(data.describe())

print("\nHighest Vehicle Count:")
print(data["Vehicle_Count"].max())

print("\nLowest Average Speed:")
print(data["Average_Speed"].min())

print("\nCongestion Distribution:")
print(data["Congestion"].value_counts())