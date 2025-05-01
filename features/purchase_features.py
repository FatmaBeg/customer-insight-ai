import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class PurchaseFeatureEngineer:
    def __init__(self, raw_df):
        """
        Initialize the feature engineer with raw data.
        
        Args:
            raw_df (pd.DataFrame): Raw data from get_purchase_raw_data()
        """
        self.raw_df = raw_df.copy()
        
    def aggregate_category_spending(self):
        """
        Compute total spending by customer and category, then pivot to wide format.
        
        Returns:
            DataFrame with customer_id as index and category spending as columns
        """
        # Pivot the data to get category spending per customer
        category_spending = self.raw_df.pivot_table(
            index='customer_id',
            columns='category_name',
            values='total_spent',
            fill_value=0
        )
        
        # Add total spending across all categories
        category_spending['total_spending'] = category_spending.sum(axis=1)
        
        # Normalize category spending by total spending
        for col in category_spending.columns:
            if col != 'total_spending':
                category_spending[f'{col}_ratio'] = (
                    category_spending[col] / category_spending['total_spending']
                ).fillna(0)
        
        return category_spending
        
    def encode_binary(self, df):
        """
        Generate binary target from category interaction.
        A customer is considered interested if:
        - They have spent above average in any category
        - OR they have spent more than 20% of their total spending in any category
        
        Args:
            df (pd.DataFrame): DataFrame with category spending features
            
        Returns:
            tuple of (X, y) where:
            - X: DataFrame with spending features
            - y: Series with binary target
        """
        # Get category columns (excluding total_spending and ratio columns)
        category_cols = [col for col in df.columns 
                        if not col.endswith('_ratio') and col != 'total_spending']
        
        # Create binary target based on spending patterns
        y = pd.Series(0, index=df.index)
        
        for category in category_cols:
            # Calculate average spending in category
            avg_spending = df[category].mean()
            
            # Calculate spending ratio
            spending_ratio = df[category] / df['total_spending']
            
            # Update target if conditions met
            y |= (
                (df[category] > avg_spending) | 
                (spending_ratio > 0.2)
            ).astype(int)
        
        # Prepare features (X)
        X = df.copy()
        
        return X, y
        
    def transform(self):
        """
        Combine all feature engineering steps and return X, y for model training.
        
        Returns:
            tuple of (X, y) where:
            - X: DataFrame with spending features
            - y: Series with binary target
        """
        # Get category spending features
        category_features = self.aggregate_category_spending()
        
        # Generate binary target
        X, y = self.encode_binary(category_features)
        
        return X, y
