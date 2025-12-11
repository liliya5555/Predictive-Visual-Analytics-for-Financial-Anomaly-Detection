
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load preprocessed data
df = pd.read_csv("processed_financial_data.csv")

# Features (excluding label and id)
features = df.drop(columns=["Transaction ID", "Timestamp", "Is Anomaly"])

# Isolation Forest
model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
df['Predicted Anomaly'] = model.fit_predict(features)

# Convert prediction to binary anomaly
df['Predicted Anomaly'] = df['Predicted Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Save model
joblib.dump(model, "isolation_forest_model.pkl")

# Save predictions
df.to_csv("anomaly_predictions.csv", index=False)
