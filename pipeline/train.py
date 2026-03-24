import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pickle
import os

df = pd.read_csv("../data/water_potability.csv")
df = df.fillna(df.mean())

features = df.drop(columns=["Potability"])
labels = df["Potability"].map({0: -1, 1: 1})

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

print("=" * 50)
print("MODEL COMPARISON: Isolation Forest vs LOF")
print("=" * 50)

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.4,
    random_state=42
)
iso_preds = iso_forest.fit_predict(scaled)

iso_precision = precision_score(labels, iso_preds)
iso_recall = recall_score(labels, iso_preds)
iso_f1 = f1_score(labels, iso_preds)

print("\n--- Isolation Forest ---")
print(f"Precision: {iso_precision:.4f}")
print(f"Recall:    {iso_recall:.4f}")
print(f"F1 Score:  {iso_f1:.4f}")
print(classification_report(labels, iso_preds, target_names=["Anomaly", "Normal"]))

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.4
)
lof_preds = lof.fit_predict(scaled)

lof_precision = precision_score(labels, lof_preds)
lof_recall = recall_score(labels, lof_preds)
lof_f1 = f1_score(labels, lof_preds)

print("\n--- Local Outlier Factor ---")
print(f"Precision: {lof_precision:.4f}")
print(f"Recall:    {lof_recall:.4f}")
print(f"F1 Score:  {lof_f1:.4f}")
print(classification_report(labels, lof_preds, target_names=["Anomaly", "Normal"]))

print("\n" + "=" * 50)
if iso_f1 >= lof_f1:
    best_model = iso_forest
    best_name = "Isolation Forest"
    best_f1 = iso_f1
else:
    best_model = None
    best_name = "Local Outlier Factor"
    best_f1 = lof_f1

print(f"Winner: {best_name} (F1: {best_f1:.4f})")
print("Saving best model...")

os.makedirs("../models", exist_ok=True)
with open("../models/model.pkl", "wb") as f:
    pickle.dump(iso_forest, f)
with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

metrics = {
    "isolation_forest": {"precision": round(iso_precision, 4), "recall": round(iso_recall, 4), "f1": round(iso_f1, 4)},
    "local_outlier_factor": {"precision": round(lof_precision, 4), "recall": round(lof_recall, 4), "f1": round(lof_f1, 4)},
    "winner": best_name
}
import json
with open("../models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done! Model, scaler and metrics saved.")