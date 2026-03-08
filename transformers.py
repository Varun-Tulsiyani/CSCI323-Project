import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import pdist
from itertools import combinations


class DistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, x_columns=range(16), y_columns=range(16, 32)):
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.pairs = list(combinations(range(1, 17), 2))
        self.column_names = [f"d_{i}_{j}" for i, j in self.pairs]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X.iloc[:, self.x_columns].values
        y = X.iloc[:, self.y_columns].values
        coords = np.stack((x, y), axis=2)
        distances = np.array([pdist(sample, metric='euclidean') for sample in coords])
        return pd.concat([
            X.reset_index(drop=True),
            pd.DataFrame(distances, columns=self.column_names, index=X.index).reset_index(drop=True),
        ], axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = []
        return np.array(list(input_features) + self.column_names)
