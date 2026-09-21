import pandas as pd
import matplotlib.pyplot as plt

# Load the traffic dataset
data = pd.read_csv("data/traffic.csv")

# Create a graph
plt.figure(figsize=(10, 5))

plt.plot(data["Time"], data["Vehicle_Count"], marker="o")

plt.title("Traffic Volume by Time")
plt.xlabel("Time")
plt.ylabel("Number of Vehicles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()