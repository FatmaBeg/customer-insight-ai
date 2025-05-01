import os
import joblib
import pickle
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.models import Model

def save_model(model: Model, path: str) -> None:
    """
    Save a Keras model to the specified path.
    
    Args:
        model: Keras model to save
        path: Path where to save the model (should end with .h5)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")

def load_model(path: str) -> Model:
    """
    Load a Keras model from the specified path.
    
    Args:
        path: Path to the saved model (.h5 file)
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return keras_load_model(path)

def save_scaler(scaler, path: str) -> None:
    """
    Save a scaler object using joblib.
    
    Args:
        scaler: Scaler object to save (e.g., StandardScaler)
        path: Path where to save the scaler (should end with .pkl)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")

def load_scaler(path: str):
    """
    Load a scaler object from the specified path.
    
    Args:
        path: Path to the saved scaler (.pkl file)
        
    Returns:
        Loaded scaler object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found at {path}")
    return joblib.load(path)

def save_shap_explainer(explainer, path: str) -> None:
    """
    Save a SHAP explainer object using joblib.
    
    Args:
        explainer: SHAP explainer object to save
        path: Path where to save the explainer (should end with .pkl)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(explainer, path)
    print(f"SHAP explainer saved to {path}")

def load_shap_explainer(path: str):
    """
    Load a SHAP explainer object from the specified path.
    
    Args:
        path: Path to the saved explainer (.pkl file)
        
    Returns:
        Loaded SHAP explainer object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SHAP explainer file not found at {path}")
    return joblib.load(path)

def save_object(obj, path: str) -> None:
    """
    Generic function to save any Python object using joblib.
    
    Args:
        obj: Object to save
        path: Path where to save the object (should end with .pkl)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"Object saved to {path}")

def load_object(path: str):
    """
    Generic function to load any Python object using joblib.
    
    Args:
        path: Path to the saved object (.pkl file)
        
    Returns:
        Loaded object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Object file not found at {path}")
    return joblib.load(path)
