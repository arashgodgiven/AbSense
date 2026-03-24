from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import pickle
import numpy as np
import os
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "models", "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

client = MongoClient("mongodb://localhost:27017/")
db = client["absense"]
collection = db["readings"]

app = FastAPI(title="AbSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SensorReading(BaseModel):
    ph: float
    hardness: float
    solids: float
    chloramines: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float

@app.get("/")
def root():
    return {"status": "AbSense API is running"}

@app.post("/predict")
def predict(reading: SensorReading):
    data = np.array([[
        reading.ph,
        reading.hardness,
        reading.solids,
        reading.chloramines,
        reading.sulfate,
        reading.conductivity,
        reading.organic_carbon,
        reading.trihalomethanes,
        reading.turbidity
    ]])
    scaled = scaler.transform(data)

    prediction = model.predict(scaled)[0]
    score = model.decision_function(scaled)[0]
    is_anomaly = bool(prediction == -1)

    document = {
        "timestamp": datetime.utcnow(),
        "reading": reading.dict(),
        "anomaly": is_anomaly,
        "score": round(float(score), 4),
        "status": "ANOMALY DETECTED" if is_anomaly else "Normal"
    }
    collection.insert_one(document)

    return {
        "anomaly": is_anomaly,
        "score": round(float(score), 4),
        "status": "ANOMALY DETECTED" if is_anomaly else "Normal"
    }

@app.get("/anomalies")
def get_anomalies():
    results = list(collection.find(
        {"anomaly": True},
        {"_id": 0}
    ).sort("timestamp", -1).limit(20))
    return results

@app.post("/retrain")
def retrain():
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, f1_score
    import pickle
    import json

    # Load and prepare data
    df = pd.read_csv("../data/water_potability.csv")
    df = df.fillna(df.mean())
    features = df.drop(columns=["Potability"])
    labels = df["Potability"].map({0: -1, 1: 1})

    # Scale
    new_scaler = StandardScaler()
    scaled = new_scaler.fit_transform(features)

    # Train both models
    iso = IsolationForest(n_estimators=100, contamination=0.4, random_state=42)
    iso_preds = iso.fit_predict(scaled)
    iso_f1 = f1_score(labels, iso_preds)
    iso_precision = precision_score(labels, iso_preds)
    iso_recall = recall_score(labels, iso_preds)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.4)
    lof_preds = lof.fit_predict(scaled)
    lof_f1 = f1_score(labels, lof_preds)
    lof_precision = precision_score(labels, lof_preds)
    lof_recall = recall_score(labels, lof_preds)

    # Save best model (always ISO for serving reasons)
    winner = "Isolation Forest" if iso_f1 >= lof_f1 else "Local Outlier Factor"

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(BASE_DIR, "models", "model.pkl"), "wb") as f:
        pickle.dump(iso, f)
    with open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "wb") as f:
        pickle.dump(new_scaler, f)

    # Update global model and scaler
    global model, scaler
    model = iso
    scaler = new_scaler

    # Save and return metrics
    metrics = {
        "isolation_forest": {
            "precision": round(iso_precision, 4),
            "recall": round(iso_recall, 4),
            "f1": round(iso_f1, 4)
        },
        "local_outlier_factor": {
            "precision": round(lof_precision, 4),
            "recall": round(lof_recall, 4),
            "f1": round(lof_f1, 4)
        },
        "winner": winner,
        "model_in_use": "Isolation Forest",
        "reason": "LOF cannot score new individual samples at inference time"
    }

    with open(os.path.join(BASE_DIR, "models", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {"status": "Retrained successfully", "metrics": metrics}

@app.get("/metrics")
def get_metrics():
    import json
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    if not os.path.exists(metrics_path):
        return {"error": "No metrics found, run /retrain first"}
    with open(metrics_path) as f:
        return json.load(f)