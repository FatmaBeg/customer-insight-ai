import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, hamming_loss, f1_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from data.queries import get_purchase_raw_data
from features.purchase_features import PurchaseFeatureEngineer

def create_mlp_model(input_dim: int, output_dim: int) -> Model:
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
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

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = create_mlp_model(input_dim=X_train.shape[1], output_dim=y.shape[1])
        
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
        
        # 6. Model Evaluation
        print("\nModel Evaluation:")
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.4).astype(int)

        print("Hamming Loss:", hamming_loss(y_test, y_pred_binary))
        print("Micro F1 Score:", f1_score(y_test, y_pred_binary, average='micro'))
        print("Macro F1 Score:", f1_score(y_test, y_pred_binary, average='macro'))
        print("Sample-based accuracy (exact match):", (y_test.values == y_pred_binary).mean())
        
        print("\nTraining Summary:")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Number of epochs trained: {len(history.history['loss'])}")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    pass