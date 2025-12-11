
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load and sort dataset
df = pd.read_csv("processed_financial_data.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp')

# Select features for LSTM
data = df[['Amount', 'Risk Score']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare sequences
X, y = [], []
window_size = 10
for i in range(len(data_scaled) - window_size):
    X.append(data_scaled[i:i+window_size])
    y.append(data_scaled[i+window_size][1])
X, y = np.array(X), np.array(y)

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=5, batch_size=64)

# Save model
model.save("lstm_risk_prediction_model.h5")
