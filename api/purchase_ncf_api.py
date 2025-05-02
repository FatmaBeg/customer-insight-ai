from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "trained_models", "purchase_ncf_model.h5")
USER_ENCODER_PATH = os.path.join(BASE_DIR, "..", "models", "trained_models", "purchase_ncf_user_encoder.pkl")
ITEM_ENCODER_PATH = os.path.join(BASE_DIR, "..", "models", "trained_models", "purchase_ncf_item_encoder.pkl")


model = None
user_encoder = None
item_encoder = None

class PurchaseNCFRequest(BaseModel):
    customer_id: str
    top_k: int = 3

class PurchaseNCFResponse(BaseModel):
    customer_id: str
    top_categories: List[str]

def load_ncf_model():
    global model, user_encoder, item_encoder
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("NCF model file not found")
        model = load_model(MODEL_PATH)

    if user_encoder is None:
        with open(USER_ENCODER_PATH, 'rb') as f:
            user_encoder = pickle.load(f)

    if item_encoder is None:
        with open(ITEM_ENCODER_PATH, 'rb') as f:
            item_encoder = pickle.load(f)

@router.post("/predict/purchase-ncf", response_model=PurchaseNCFResponse)
def recommend_categories(request: PurchaseNCFRequest):
    load_ncf_model()

    if request.customer_id not in user_encoder.classes_:
        raise HTTPException(status_code=404, detail="Customer ID not found in training data")

    customer_idx = user_encoder.transform([request.customer_id])[0]
    all_category_indices = np.arange(len(item_encoder.classes_))
    customer_indices = np.full_like(all_category_indices, customer_idx)

    # Predict scores for all categories
    predictions = model.predict([customer_indices, all_category_indices], verbose=0).flatten()
    top_k_indices = predictions.argsort()[::-1][:request.top_k]
    top_categories = item_encoder.inverse_transform(top_k_indices)

    return PurchaseNCFResponse(
        customer_id=request.customer_id,
        top_categories=top_categories.tolist()
    )
