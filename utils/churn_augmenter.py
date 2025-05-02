import pandas as pd

class ChurnAugmenter:
    @staticmethod
    def oversample_churners(X: pd.DataFrame, y: pd.Series, multiplier: int = 3):
        churn_X = X[y == 1]
        churn_y = y[y == 1]

        # Indexleri sıfırlayıp hizalı şekilde çoğalt
        X_aug = pd.concat([X] + [churn_X] * multiplier, ignore_index=True)
        y_aug = pd.concat([y] + [churn_y] * multiplier, ignore_index=True)

        return X_aug, y_aug