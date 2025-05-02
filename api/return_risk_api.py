from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import shap
from tensorflow.keras.models import load_model
from typing import List, Dict, Any
import os
from models.utils import load_shap_explainer

app = FastAPI(title="Return Risk Prediction API")

# Model and scaler paths
MODEL_PATH = "models/trained_models/return_risk_model.h5"
SCALER_PATH = "models/trained_models/return_risk_scaler.pkl"

# Initialize model and scaler as None
model = None
scaler = None

def load_models():
    """Load model and scaler if they exist."""
    global model, scaler
    if model is None and scaler is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                model = load_model(MODEL_PATH)
                with open(SCALER_PATH, 'rb') as f:
                    scaler = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load model or scaler: {str(e)}")
        else:
            raise RuntimeError("Model or scaler files not found. Please train the models first.")

class ReturnRiskFeatures(BaseModel):
    """Input features for return risk prediction."""
    discount: float = Field(..., ge=0, le=1, description="Discount percentage (0-1)")
    quantity: int = Field(..., gt=0, description="Quantity of items")
    price: float = Field(..., gt=0, description="Unit price")

class ReturnRiskPrediction(BaseModel):
    """Return risk prediction response."""
    return_risk: float = Field(..., description="Probability of return")
    confidence: float = Field(..., description="Confidence in prediction")
    top_features: List[Dict[str, str]] = Field(..., description="Top 3 contributing features with explanations")

def calculate_confidence(probability: float) -> float:
    """
    Calculate confidence based on how far the probability is from 0.5.
    
    Args:
        probability: Predicted probability
        
    Returns:
        Confidence score between 0 and 1
    """
    return 2 * abs(probability - 0.5)

def get_shap_explanation(features: np.ndarray) -> List[Dict[str, str]]:
    """
    Get SHAP explanation for the prediction.
    
    Args:
        features: Scaled input features
        
    Returns:
        List of dictionaries containing feature explanations
    """
    # Get SHAP values
    shap_values = shap_explainer.shap_values(features)
    
    # Get feature names and values
    feature_names = shap_explainer.feature_names
    feature_values = features[0]
    
    # Create explanation for each feature
    explanations = []
    for i, (name, value, shap) in enumerate(zip(feature_names, feature_values, shap_values[0])):
        if abs(shap) > 0.01:  # Only include significant contributions
            explanation = {
                "feature": name,
                "value": float(value),
                "impact": float(shap),
                "direction": "increases" if shap > 0 else "decreases"
            }
            explanations.append(explanation)
    
    # Sort by absolute impact and get top 3
    explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return explanations[:3]

@app.post("/predict/return-risk", response_model=ReturnRiskPrediction)
async def predict_return_risk(features: ReturnRiskFeatures) -> Dict:
    """
    Predict return risk probability.
    
    Args:
        features: Order features for prediction
        
    Returns:
        Dictionary containing return risk probability, confidence, and top features
    """
    try:
        # Load models if not already loaded
        load_models()
        
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not ready. Please try again later.")
            
        # Convert features to numpy array
        feature_array = np.array([
            features.discount,
            features.quantity,
            features.price
        ]).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        return_risk = float(model.predict(scaled_features)[0][0])
        
        # Calculate confidence
        confidence = calculate_confidence(return_risk)
        
        # Get SHAP explanation
        top_features = get_shap_explanation(scaled_features)
        
        return {
            "return_risk": return_risk,
            "confidence": confidence,
            "top_features": top_features
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    try:
        load_models()
        return {"status": "healthy", "model_loaded": True}
    except HTTPException:
        return {"status": "healthy", "model_loaded": False, "message": "Model not trained yet"}
