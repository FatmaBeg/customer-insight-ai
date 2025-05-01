import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from data.queries import get_churn_raw_data
from features.churn_features import ChurnFeatureEngineer
from explainability.shap_explainer import SHAPExplainer

def create_mlp_model(input_dim):
    """
    Create a Multi-Layer Perceptron model for churn prediction.
    
    Args:
        input_dim (int): Number of input features
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main(engine):
    """
    Main function to train the churn prediction model.
    
    Args:
        engine: SQLAlchemy engine for database connection
    """
    # Create output directory if it doesn't exist
    os.makedirs('models/trained_models', exist_ok=True)
    
    # 1. Load raw data
    print("Loading raw data...")
    raw_data = get_churn_raw_data(engine)
    
    # 2. Extract features and pseudo-labels
    print("Extracting features...")
    feature_engineer = ChurnFeatureEngineer(raw_data)
    features = feature_engineer.transform()
    
    # Prepare X and y
    X = features.drop(['customer_id', 'will_churn'], axis=1)
    y = features['will_churn']
    
    # 3. Standardize features and split data
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate class weights
    class_weights = {
        0: len(y_train) / (2 * (len(y_train) - y_train.sum())),
        1: len(y_train) / (2 * y_train.sum())
    }
    
    # 4. Train model
    print("Training model...")
    model = create_mlp_model(input_dim=X_train.shape[1])
    
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
        class_weight=class_weights,
        verbose=1
    )
    
    # 5. Save model and scaler
    print("Saving model and scaler...")
    model.save('models/trained_models/churn_model.h5')
    with open('models/trained_models/churn_scaler.pkl', 'wb') as f:
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
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nTop 3 Contributing Features:")
    for feature, importance in top_features:
        # Ensure importance is a float
        importance_float = float(importance)
        print(f"{feature}: {importance_float:.4f}")
    
    print("\nTraining Summary:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Number of epochs trained: {len(history.history['loss'])}")

if __name__ == "__main__":
    # Engine will be passed from run.py
    pass
