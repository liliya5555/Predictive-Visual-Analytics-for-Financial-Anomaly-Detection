
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_excel("financial_anomaly_dataset.xlsx")

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Normalize amount and risk score
scaler = MinMaxScaler()
df[['Amount', 'Risk Score']] = scaler.fit_transform(df[['Amount', 'Risk Score']])

# Encode transaction type and market
df = pd.get_dummies(df, columns=['Transaction Type', 'Market'])

# Save preprocessed data
df.to_csv("processed_financial_data.csv", index=False)
