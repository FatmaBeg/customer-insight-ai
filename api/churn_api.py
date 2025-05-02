from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import shap
from tensorflow.keras.models import load_model
from typing import List, Dict, Any
import os

app = FastAPI(title="Churn Prediction API")

# Model and scaler paths
MODEL_PATH = "models/trained_models/churn_model.h5"
SCALER_PATH = "models/scalers/churn_scaler.pkl"

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

class ChurnFeatures(BaseModel):
    """Input features for churn prediction."""
    total_spent: float = Field(..., description="Total amount spent by customer")
    total_orders: int = Field(..., description="Total number of orders")
    avg_order_value: float = Field(..., description="Average order value")
    customer_tenure_days: int = Field(..., description="Number of days since first order")
    year: int = Field(..., description="Current year")
    month: int = Field(..., description="Current month")
    season: int = Field(..., description="Season (1: Winter, 2: Spring, 3: Summer, 4: Fall)")
    is_summer: int = Field(..., description="Is it summer? (1: Yes, 0: No)")

class ChurnPrediction(BaseModel):
    """Churn prediction response."""
    churn_probability: float = Field(..., description="Probability of churn")
    will_churn: bool = Field(..., description="Will the customer churn?")
    confidence: float = Field(..., description="Confidence in prediction")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")

def calculate_confidence(probability: float) -> float:
    """
    Calculate confidence based on how far the probability is from 0.5.
    
    Args:
        probability: Churn probability
        
    Returns:
        Confidence score between 0 and 1
    """
    return 1 - 2 * abs(0.5 - probability)

@app.post("/predict/churn", response_model=ChurnPrediction)
async def predict_churn(features: ChurnFeatures):
    """
    Predict churn probability for a customer.
    
    Args:
        features: Customer features
        
    Returns:
        Churn prediction with probability and confidence
    """
    try:
        # Load models if not already loaded
        load_models()
        
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not ready. Please try again later.")
            
        # Convert features to numpy array
        feature_array = np.array([
            features.total_spent,
            features.total_orders,
            features.avg_order_value,
            features.customer_tenure_days,
            features.year,
            features.month,
            features.season,
            features.is_summer
        ]).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        churn_probability = float(model.predict(scaled_features)[0][0])
        will_churn = churn_probability > 0.5
        
        # Calculate confidence
        confidence = calculate_confidence(churn_probability)
        
        # Create SHAP explainer and get feature importance
        explainer = shap.KernelExplainer(
            lambda x: model.predict(x),
            shap.sample(scaled_features, 50)
        )
        shap_values = explainer.shap_values(scaled_features)
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values[0])
        
        # Get feature importance scores
        feature_names = [
            "total_spent", "total_orders", "avg_order_value", "customer_tenure_days",
            "year", "month", "season", "is_summer"
        ]
        feature_importance = {
            name: float(abs(value))
            for name, value in zip(feature_names, shap_values[0])
        }
        
        return ChurnPrediction(
            churn_probability=churn_probability,
            will_churn=will_churn,
            confidence=confidence,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status of the API
    """
    return {"status": "healthy"}
