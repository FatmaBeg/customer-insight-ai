from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import shap
from tensorflow.keras.models import load_model
from typing import List, Dict, Any
import os

app = FastAPI(title="Purchase Prediction API")

# Model and scaler paths
MODEL_PATH = "models/trained_models/purchase_model.h5"
SCALER_PATH = "models/scalers/purchase_scaler.pkl"

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

class CategorySpending(BaseModel):
    """Input features for product recommendation."""
    category_spending: Dict[str, float] = Field(
        ...,
        description="Dictionary of category names and their corresponding spending amounts"
    )

class PurchasePrediction(BaseModel):
    """Product recommendation response."""
    top_predictions: List[Dict[str, float]] = Field(
        ...,
        description="List of top 3 predicted categories with their probabilities"
    )

def get_top_predictions(probabilities: np.ndarray, k: int = 3) -> List[Dict[str, float]]:
    """
    Get top k predictions with their probabilities.
    
    Args:
        probabilities: Array of predicted probabilities
        k: Number of top predictions to return
        
    Returns:
        List of dictionaries containing category names and probabilities
    """
    # Get indices of top k probabilities
    top_indices = np.argsort(probabilities[0])[-k:][::-1]
    
    # Create predictions with category names and probabilities
    predictions = []
    for idx in top_indices:
        predictions.append({
            "category": category_names[idx],
            "probability": float(probabilities[0][idx])
        })
    
    return predictions

@app.post("/predict/purchase", response_model=PurchasePrediction)
async def predict_purchase(spending: CategorySpending) -> Dict:
    """
    Predict top product categories based on category-wise spending.
    
    Args:
        spending: Dictionary of category names and spending amounts
        
    Returns:
        Dictionary containing top predicted categories and their probabilities
    """
    try:
        # Load models if not already loaded
        load_models()
        
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not ready. Please try again later.")
            
        # Create feature array from category spending
        feature_array = np.zeros(len(category_names))
        for category, amount in spending.category_spending.items():
            if category in category_names:
                idx = category_names.index(category)
                feature_array[idx] = amount
        
        # Reshape for prediction
        feature_array = feature_array.reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        probabilities = model.predict(scaled_features)
        
        # Get top predictions
        top_predictions = get_top_predictions(probabilities)
        
        return {
            "top_predictions": top_predictions
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
