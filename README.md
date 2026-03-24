# AbSense

Water Quality Anomaly Detector — powered by Isolation Forest

## What it does
AbSense analyzes water sensor readings (pH, turbidity, hardness, etc.) and flags
anomalies using a machine learning model trained on real-world water quality data.
Unusual readings are detected, scored, and logged for trend analysis.

## About the name
AbSense combines **"Ab"** (آب) — the Persian/Farsi word for water — with **"Sense"**, 
reflecting both the sensor-driven nature of the project and the idea of making sense 
of water quality data. The name also carries a subtle double meaning: detecting the 
*absence* of normal patterns in sensor readings.

## Tech Stack
- **ML Model** — Isolation Forest (scikit-learn) trained on the Water Potability dataset
- **Backend** — Python, FastAPI, REST API
- **Database** — MongoDB (logs all readings and anomaly scores with timestamps)
- **Frontend** — React, Recharts (live anomaly score history chart)
- **DevOps** — Docker

## How to run

### API (local)
```bash
cd api
python -m uvicorn main:app --reload
```

### API (Docker)
```bash
docker build -t absense .
docker run -p 8000:8000 absense
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## API Endpoints
- `POST /predict` — submit a sensor reading, returns anomaly status and score
- `GET /anomalies` — returns the last 20 flagged anomalies from MongoDB

## Dataset
[Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
— 3,276 real water quality samples with 9 sensor features.
