import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from fastapi import FastAPI
from typing import Any, Dict

# Load model
model = joblib.load("models/xgb_fraud.joblib")

# Get exact feature names model expects
MODEL_FEATURES = model.feature_names_in_.tolist()

app = FastAPI(title="FraudShield API", version="1.0.0")


@app.get("/")
def root():
    return {"message": "FraudShield API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def get_features():
    return {"features": MODEL_FEATURES, "count": len(MODEL_FEATURES)}


@app.post("/predict")
def predict(transaction: Dict[str, Any]):
    # Create dataframe with all model features, fill missing with 0
    data = pd.DataFrame([transaction])

    for feature in MODEL_FEATURES:
        if feature not in data.columns:
            data[feature] = 0

    # Keep only model features in correct order
    data = data[MODEL_FEATURES]

    proba = model.predict_proba(data)[:, 1][0]
    threshold = 0.3
    is_fraud = bool(proba >= threshold)

    if proba >= 0.7:
        risk_level = "HIGH"
    elif proba >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(float(proba), 4),
        "risk_level": risk_level
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)