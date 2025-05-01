import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from data.queries import get_purchase_raw_data
from features.purchase_features import PurchaseFeatureEngineer
from explainability.shap_explainer import SHAPExplainer

def create_mlp_model(input_dim: int) -> Model:
    """
    Create a Multi-Layer Perceptron model for purchase prediction.
    
    Args:
        input_dim: Number of input features
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',  # Binary crossentropy for binary classification
        metrics=['accuracy']
    )
    
    return model

def main(engine):
    """
    Main function to train the purchase prediction model.
    
    Args:
        engine: SQLAlchemy engine for database connection
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs('models/trained_models', exist_ok=True)
        
        # 1. Load raw data
        print("Loading raw data...")
        raw_data = get_purchase_raw_data(engine)
        
        # 2. Extract features and target
        print("Extracting features...")
        feature_engineer = PurchaseFeatureEngineer(raw_data)
        X, y = feature_engineer.transform()
        
        # 3. Standardize features and split data
        print("Preprocessing data...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 4. Train model
        print("Training model...")
        model = create_mlp_model(X_train.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 5. Save model and scaler
        print("Saving model and scaler...")
        model.save('models/trained_models/purchase_model.h5')
        with open('models/trained_models/purchase_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # 6. Analyze feature importance with SHAP
        print("\nAnalyzing feature importance...")
        feature_names = X.columns.tolist()
        
        # Create and use SHAP explainer
        shap_explainer = SHAPExplainer(model)
        shap_explainer.create_explainer(X_train)
        shap_values, top_features = shap_explainer.analyze_feature_importance(X_train, feature_names)
        
        # Print metrics and feature importance
        print("\nModel Evaluation:")
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        print("\nClassification Report:")
        print(f"Accuracy: {np.mean(y_test == y_pred_binary.flatten()):.4f}")
        
        print("\nTop 3 Contributing Features:")
        for feature, importance in top_features:
            # Ensure importance is a float
            importance_float = float(importance)
            print(f"{feature}: {importance_float:.4f}")
        
        print("\nTraining Summary:")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Number of epochs trained: {len(history.history['loss'])}")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    # Engine will be passed from run.py
    pass
