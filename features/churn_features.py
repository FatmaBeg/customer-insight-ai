import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ChurnFeatureEngineer:
    def __init__(self, raw_df):
        """
        Initialize the feature engineer with raw data.
        
        Args:
            raw_df (pd.DataFrame): Raw data from get_churn_raw_data()
        """
        self.raw_df = raw_df.copy()
        self.raw_df['order_date'] = pd.to_datetime(self.raw_df['order_date'])
        
    def add_temporal_features(self):
        """
        Extract temporal features from order_date:
        - year
        - month
        - season (1: Winter, 2: Spring, 3: Summer, 4: Fall)
        - is_summer (binary)
        """
        df = self.raw_df.copy()
        
        # Extract year and month
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        
        # Extract season
        df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3)
        
        # Create is_summer feature
        df['is_summer'] = (df['month'].isin([6, 7, 8])).astype(int)
        
        return df
        
    def add_churn_target(self, df):
        """
        Label customers as churned (1) if no order placed within 6 months after last order.
        
        Args:
            df (pd.DataFrame): DataFrame with temporal features
            
        Returns:
            pd.DataFrame with will_churn column
        """
        # Get last order date for each customer
        last_orders = df.groupby('customer_id')['order_date'].max().reset_index()
        last_orders.columns = ['customer_id', 'last_order_date']
        
        # Calculate churn status
        cutoff_date = df['order_date'].max() - timedelta(days=180)  # 6 months
        last_orders['will_churn'] = (last_orders['last_order_date'] < cutoff_date).astype(int)
        
        # Merge back to original dataframe
        df = df.merge(last_orders[['customer_id', 'will_churn']], on='customer_id', how='left')
        
        return df
        
    def aggregate_customer_metrics(self, df):
        """
        Calculate customer-level metrics:
        - total_spent: sum of all order values
        - total_orders: count of unique orders
        - avg_order_value: total_spent / total_orders
        """
        # Calculate metrics
        metrics = df.groupby('customer_id').agg({
            'total_spent': 'sum',
            'order_id': 'nunique',
            'order_date': ['min', 'max']  # for calculating customer tenure
        }).reset_index()
        
        # Flatten multi-index columns
        metrics.columns = ['customer_id', 'total_spent', 'total_orders', 'first_order_date', 'last_order_date']
        
        # Calculate average order value
        metrics['avg_order_value'] = metrics['total_spent'] / metrics['total_orders']
        
        # Calculate customer tenure in days
        metrics['customer_tenure_days'] = (metrics['last_order_date'] - metrics['first_order_date']).dt.days
        
        # Drop date columns as we don't need them anymore
        metrics = metrics.drop(['first_order_date', 'last_order_date'], axis=1)
        
        return metrics
        
    def transform(self):
        """
        Combine all feature engineering steps and return final features.
        
        Returns:
            pd.DataFrame with all features and will_churn target
        """
        # Add temporal features
        df = self.add_temporal_features()
        
        # Add churn target
        df = self.add_churn_target(df)
        
        # Get customer metrics
        metrics = self.aggregate_customer_metrics(df)
        
        # Get the most recent temporal features for each customer
        latest_features = df.sort_values('order_date').groupby('customer_id').last()[
            ['year', 'month', 'season', 'is_summer', 'will_churn']
        ].reset_index()
        
        # Combine all features
        final_features = metrics.merge(latest_features, on='customer_id')
        
        return final_features
