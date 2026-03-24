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