# AbSense
Water Quality Anomaly Detector — unsupervised anomaly detection with Isolation Forest

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
- **ML** — Isolation Forest & Local Outlier Factor (scikit-learn)
- **Backend** — Python, FastAPI, REST API
- **Database** — MongoDB (logs all readings and anomaly scores with timestamps)
- **Frontend** — React, Recharts (live anomaly score history chart)
- **DevOps** — Docker

## Model Comparison
Two unsupervised anomaly detection algorithms were trained and evaluated against the
dataset's ground truth potability labels. Non-potable samples (Potability=0) were treated
as anomalies.

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| Isolation Forest | 0.3622 | 0.5571 | **0.4390** |
| Local Outlier Factor | 0.3596 | 0.5532 | 0.4359 |

**Winner: Isolation Forest**

The scores are comparable between both models, suggesting the dataset itself is the
limiting factor rather than the algorithm — water potability is genuinely difficult to
infer from these 9 sensor features alone. This is expected behavior for unsupervised
anomaly detection.

**Why Isolation Forest was chosen for serving:** LOF requires the full training dataset
to be present at inference time, making it unsuitable for a REST API that scores
individual samples. Isolation Forest can score new samples independently after training.

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
### Retrain the model
```bash
cd pipeline
python train.py
```
Or via the API:
```bash
curl -X POST http://127.0.0.1:8000/retrain
```

## API Endpoints
- `POST /predict` — submit a sensor reading, returns anomaly status and score
- `GET /anomalies` — returns the last 20 flagged anomalies from MongoDB
- `GET /metrics` — returns current model evaluation metrics
- `POST /retrain` — retrains both models on demand, returns fresh metrics

## Dataset
[Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
— 3,276 real water quality samples with 9 sensor features.
