import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from data.queries import get_purchase_raw_data
from features.purchase_features_ncf import PurchaseNCFFeatureEngineer
from training.ncf_model import create_ncf_model


def main(engine):
    """
    Main function to train the NCF model for purchase prediction.
    """
    print("Loading raw data...")
    raw_df = get_purchase_raw_data(engine)

    print("Generating features for NCF...")
    feature_engineer = PurchaseNCFFeatureEngineer(raw_df)
    df, user_encoder, item_encoder = feature_engineer.transform()

    print("Splitting data...")
    X = df[['customer_idx', 'category_idx']].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_users = df['customer_idx'].nunique()
    num_items = df['category_idx'].nunique()

    print("Building model...")
    model = create_ncf_model(num_users, num_items)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )

    print("Training model...")
    model.fit(
        [X_train[:, 0], X_train[:, 1]], y_train,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
        batch_size=64,
        epochs=70,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Saving model and encoders...")
    os.makedirs('models/trained_models', exist_ok=True)
    model.save('models/trained_models/purchase_ncf_model.h5')

    with open('models/trained_models/purchase_ncf_user_encoder.pkl', 'wb') as f:
        pickle.dump(user_encoder, f)

    with open('models/trained_models/purchase_ncf_item_encoder.pkl', 'wb') as f:
        pickle.dump(item_encoder, f)

    print("Done.")

if __name__ == "__main__":
    # Engine should be passed externally in production
    pass