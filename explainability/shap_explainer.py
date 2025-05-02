import numpy as np
import shap
from typing import Tuple, List, Any

class SHAPExplainer:
    """
    A class to handle SHAP value calculations and feature importance analysis.
    """
    
    def __init__(self, model, background_samples: int = 50, max_samples: int = 50):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained Keras model
            background_samples: Number of samples to use for background distribution
            max_samples: Maximum number of samples to use for SHAP value calculation
        """
        self.model = model
        self.background_samples = background_samples
        self.max_samples = max_samples
        self.explainer = None
    
    def _predict_wrapper(self, x):
        """
        Wrapper function for model prediction.
        
        Args:
            x: Input data
            
        Returns:
            Model predictions
        """
        return self.model.predict(x)
    
    def create_explainer(self, X_train: np.ndarray) -> None:
        """
        Create a SHAP explainer using the training data.
        
        Args:
            X_train: Training data
        """
        background = shap.sample(X_train, self.background_samples)
        self.explainer = shap.KernelExplainer(
            self._predict_wrapper,
            background
        )
    
    def _safe_convert_to_int(self, idx):
        """
        Safely convert numpy index to Python int.
        
        Args:
            idx: Index value (can be numpy.int64, np.ndarray, etc.)
            
        Returns:
            Python int
        """
        if isinstance(idx, np.ndarray):
            if idx.size == 1:
                return int(idx.item())
            raise ValueError("Index array must have exactly one element")
        return int(idx)
    
    def analyze_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Analyze feature importance using SHAP values.
        
        Args:
            X: Data to analyze
            feature_names: List of feature names
            
        Returns:
            Tuple of (shap_values, top_features)
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        # Limit the number of samples to avoid long runtimes
        X_sample = X[:self.max_samples]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle multi-output case
        # Handle multi-output case (örneğin sigmoid çıkışlı modelde liste dönebilir)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # sadece ilk çıkışı al
        else:
            shap_values = np.array(shap_values)
         # mean absolute shap değerlerini al
        mean_shap_values = np.abs(shap_values).mean(axis=0)

        # En büyük 3 özelliği sırala
        top_indices = np.argsort(mean_shap_values)[-3:][::-1]
        top_features = [
            (feature_names[int(i)], float(mean_shap_values[int(i)]))
            for i in top_indices
        ]
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Get top 3 features using numpy's argsort
        top_indices = np.argsort(mean_shap_values)[-3:][::-1]
        
        # Create top features list
        top_features = []
        for idx in top_indices:
            try:
                safe_idx = self._safe_convert_to_int(idx)
                feature_name = feature_names[safe_idx]
                importance = float(mean_shap_values[idx])
                top_features.append((feature_name, importance))
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not process index {idx}: {str(e)}")
                continue
        
        return shap_values, top_features
    
    def explain_prediction(self, data: np.ndarray, max_samples: int = None) -> np.ndarray:
        """
        Explain model predictions using SHAP values, limited to a number of samples.
        
        Args:
            data: Input data to explain
            max_samples: Max number of rows to compute SHAP for (defaults to self.max_samples)
            
        Returns:
            SHAP values for the input data
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        # Use provided max_samples or default to class value
        max_samples = max_samples if max_samples is not None else self.max_samples
        
        # Limit the number of samples to avoid long runtimes
        data_to_explain = data[:max_samples]
        
        shap_values = self.explainer.shap_values(data_to_explain)
        
        # Handle multi-output case
        if isinstance(shap_values, list):
            # For multi-output models, average the SHAP values across outputs
            shap_values = np.mean([np.array(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.array(shap_values)
            
        return shap_values
