import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os

sdf = pd.read_csv("../data/water_potability.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

df = df.fillna(df.mean())

features = df.drop(columns=["Potability"])

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
model.fit(scaled)

predictions = model.predict(scaled)
anomaly_count = list(predictions).count(-1)
print(f"\nAnomalies detected: {anomaly_count} out of {len(df)} samples")

os.makedirs("../models", exist_ok=True)
with open("../models/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved to /models")