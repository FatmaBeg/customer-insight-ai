import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ReturnRiskFeatureEngineer:
    def __init__(self, raw_df):
        """
        Initialize the feature engineer with raw data.
        
        Args:
            raw_df (pd.DataFrame): Raw data from get_return_risk_raw_data()
        """
        self.raw_df = raw_df.copy()
        
    def add_discount_features(self):
        """
        Calculate discount-related features:
        - price_per_quantity: unit_price / quantity
        - normalized_discount: discount scaled between 0 and 1
        """
        df = self.raw_df.copy()
        
        # Calculate price per quantity
        df['price_per_quantity'] = df['unit_price'] / df['quantity']
        
        # Normalize discount using MinMaxScaler
        scaler = MinMaxScaler()
        df['normalized_discount'] = scaler.fit_transform(df[['discount']])
        
        return df
        
    def add_pseudo_labels(self, df, price_threshold=100):
        """
        Generate pseudo labels for return risk:
        - Label = 1 if discount > 0.2 and total_price < threshold
        - This is a simplified heuristic for demonstration
        
        Args:
            df (pd.DataFrame): DataFrame with discount features
            price_threshold (float): Threshold for total_price to consider as risky
            
        Returns:
            pd.DataFrame with return_risk column
        """
        # Create pseudo labels based on discount and total price
        df['return_risk'] = (
            (df['discount'] > 0.2) & 
            (df['total_price'] < price_threshold)
        ).astype(int)
        
        return df
        
    def transform(self, price_threshold=100):
        """
        Combine all feature engineering steps and return X, y for model training.
        
        Args:
            price_threshold (float): Threshold for total_price in pseudo labels
            
        Returns:
            tuple of (X, y) where:
            - X: DataFrame with features
            - y: Series with return_risk labels
        """
        # Add discount features
        df = self.add_discount_features()
        
        # Add pseudo labels
        df = self.add_pseudo_labels(df, price_threshold)
        
        # Select features for X
        X = df[[
            'discount',
            'normalized_discount',
            'unit_price',
            'quantity',
            'price_per_quantity',
            'total_price'
        ]]
        
        # Get target variable
        y = df['return_risk']
        
        return X, y
