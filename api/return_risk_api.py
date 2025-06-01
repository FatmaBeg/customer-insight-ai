from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from typing import List, Dict

app = FastAPI(title="Return Risk Prediction API")

# ==== PATHS ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "trained_models", "return_risk_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "trained_models", "return_risk_scaler.pkl")
TOP_FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "trained_models", "return_risk_top_features.pkl")

# ==== GLOBALS ====
model = None
scaler = None

# ==== SCHEMAS ====
class ReturnRiskInput(BaseModel):
    discount: float = Field(..., ge=0, le=1)
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)
    

class ReturnRiskOutput(BaseModel):
    return_risk: float
    confidence: float
    top_features: List[Dict[str, str]]

# ==== HELPERS ====
def load_assets():
    global model, scaler
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model file not found.")
        model = load_model(MODEL_PATH)

    if scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise RuntimeError("Scaler file not found.")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
            
def calculate_confidence(prob: float) -> float:
    return round(2 * abs(prob - 0.5), 4)

def get_saved_top_features() -> List[Dict[str, str]]:
    SHAP_TOP_FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "trained_models", "return_risk_top_features.pkl")

    if not os.path.exists(SHAP_TOP_FEATURES_PATH):
        return []

    try:
        with open(SHAP_TOP_FEATURES_PATH, "rb") as f:
            top_features = pickle.load(f)

        # Durum 1: [(feature, impact), ...]
        if isinstance(top_features[0], tuple):
            return [
                {"feature": str(name), "impact": f"{impact:.4f}"}
                for name, impact in top_features
            ]

        # Durum 2: [{"feature": ..., "impact": ...}, ...]
        elif isinstance(top_features[0], dict):
            return top_features

        # Beklenmeyen format
        else:
            return []

    except Exception as e:
        print("SHAP top feature read error:", str(e))
        return []

# ==== ENDPOINT ====
@app.post("/predict/return-risk", response_model=ReturnRiskOutput)
def predict_risk(data: ReturnRiskInput):
    try:
        load_assets()

        X = np.array([[data.discount, data.quantity, data.price]])
        X_scaled = scaler.transform(X)
        pred = float(model.predict(X_scaled)[0][0])
        confidence = calculate_confidence(pred)
        top_features = get_saved_top_features()

        return ReturnRiskOutput(
            return_risk=pred,
            confidence=confidence,
            top_features=top_features
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/health")
def health_check():
    try:
        load_assets()
        return {"status": "healthy"}
    except:
        return {"status": "unhealthy"}