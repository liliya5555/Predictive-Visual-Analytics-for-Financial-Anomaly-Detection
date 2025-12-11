
import pandas as pd
import time
import random

# Load dataset
df = pd.read_csv("anomaly_predictions.csv")

# Simulate real-time streaming
for i in range(min(100, len(df))):
    record = df.iloc[i]
    print(f"[{record['Timestamp']}] Transaction {record['Transaction ID']} | Amount: {record['Amount']:.2f} | Risk Score: {record['Risk Score']:.3f} | Anomaly: {'YES' if record['Predicted Anomaly'] == 1 else 'NO'}")
    time.sleep(random.uniform(0.05, 0.2))  # Simulated delay
