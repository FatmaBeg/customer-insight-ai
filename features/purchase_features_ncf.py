import pandas as pd
from sklearn.preprocessing import LabelEncoder

class PurchaseNCFFeatureEngineer:
    def __init__(self, raw_df):
        """
        Args:
            raw_df (pd.DataFrame): DataFrame with columns [customer_id, category_name, total_spent]
        """
        self.raw_df = raw_df.copy()

    def encode_entities(self):
        """
        Encode customer_id and category_name into integer IDs for embedding layers.
        """
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        self.raw_df['customer_idx'] = self.user_encoder.fit_transform(self.raw_df['customer_id'])
        self.raw_df['category_idx'] = self.item_encoder.fit_transform(self.raw_df['category_name'])

    def generate_interactions(self, spending_threshold_ratio: float = 0.2):
        """
        Generate binary labels: if a customer spends > 20% of their total on a category, it's positive.
        """
        df = self.raw_df.copy()

        # Total spending per customer
        total_spent = df.groupby('customer_id')['total_spent'].transform('sum')
        df['spending_ratio'] = df['total_spent'] / total_spent

        # Binary label: interested (1) if spending_ratio > threshold
        df['label'] = (df['spending_ratio'] > spending_threshold_ratio).astype(int)
        return df[['customer_idx', 'category_idx', 'label']]

    def transform(self):
        """
        Full pipeline to get training data for NCF.
        
        Returns:
            df: DataFrame with [customer_idx, category_idx, label]
            user_encoder: fitted LabelEncoder for customers
            item_encoder: fitted LabelEncoder for categories
        """
        self.encode_entities()
        df = self.generate_interactions()
        return df, self.user_encoder, self.item_encoder
