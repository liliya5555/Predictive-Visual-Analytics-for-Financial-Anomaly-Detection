
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("anomaly_predictions.csv")

# Define fake 'region' groupings for analysis
df['Region'] = df['Sender'].apply(lambda x: 'Region A' if x[-1] in 'ABCDE' else 'Region B')

# Compare anomaly rates
region_group = df.groupby('Region')['Predicted Anomaly'].mean()

# Plot fairness analysis
plt.figure(figsize=(6, 4))
region_group.plot(kind='bar', color=['blue', 'green'])
plt.title("Anomaly Detection Rate by Region")
plt.ylabel("Anomaly Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("bias_fairness_analysis.png")
