import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class PurchaseFeatureEngineer:
    def __init__(self, raw_df):
        self.raw_df = raw_df.copy()
        
    def pivot_category_spending(self):
        """
        Pivot: customer_id x category_name → total_spent
        """
        pivot_df = self.raw_df.pivot_table(
            index='customer_id',
            columns='category_name',
            values='total_spent',
            fill_value=0
        )
        return pivot_df

    def generate_multilabel_targets(self, pivot_df, threshold_ratio=0.2):
        """
        Multi-label y oluştur. 
        Her kategori için müşteri eğer toplam harcamasının %20'sinden fazlasını harcamışsa -> o kategoriye ilgilidir.
        """
        total_spent = pivot_df.sum(axis=1)
        y = (pivot_df.T / total_spent).T > threshold_ratio
        y = y.astype(int)
        return y

    def transform(self):
        """
        Returns:
            X: DataFrame of category spending (raw)
            y: Multi-label binary targets (same columns as X)
        """
        pivot_df = self.pivot_category_spending()
        y = self.generate_multilabel_targets(pivot_df)
        print(y.sum(axis=0).sort_values())
        X = pivot_df.copy()  # X raw harcama verisi
        return X, y