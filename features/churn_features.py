import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ChurnFeatureEngineer:
    def __init__(self, raw_df):
        self.raw_df = raw_df.copy()
        self.raw_df['order_date'] = pd.to_datetime(self.raw_df['order_date'])
        self.raw_df['total_spent'] = self.raw_df['unit_price'] * self.raw_df['quantity']

    def add_churn_label(self):
        """
        Label customers as churned if no order placed in the last 6 months
        """
        last_order = self.raw_df.groupby('customer_id')['order_date'].max().reset_index()
        cutoff = self.raw_df['order_date'].max() - timedelta(days=180)
        last_order['will_churn'] = (last_order['order_date'] < cutoff).astype(int)
        return last_order[['customer_id', 'will_churn']]

    def get_temporal_features(self):
        """
        Get most recent order's temporal features
        """
        df = self.raw_df.copy()
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        latest = df.sort_values('order_date').groupby('customer_id').last().reset_index()
        return latest[['customer_id', 'year', 'month', 'season', 'is_summer']]

    def get_customer_aggregates(self):
        """
        Calculate per-customer aggregates
        """
        agg = self.raw_df.groupby('customer_id').agg(
            total_spent=pd.NamedAgg(column='total_spent', aggfunc='sum'),
            total_orders=pd.NamedAgg(column='order_id', aggfunc='nunique'),
        ).reset_index()
        agg['avg_order_value'] = agg['total_spent'] / agg['total_orders']
        return agg

    def transform(self):
        """
        Combine all features and return final DataFrame with target
        """
        agg = self.get_customer_aggregates()
        temporal = self.get_temporal_features()
        target = self.add_churn_label()
        
        final_df = agg.merge(temporal, on='customer_id')
        final_df = final_df.merge(target, on='customer_id')
        return final_df