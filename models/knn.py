from models.utils import Strategy, strategies
import pandas as pd
import numpy as np
from collections import Counter
from models.metrics import metric_functions, Metric


class KNN:
    def __init__(self, k: int = 2, *, strategy: Strategy = 'euclidean'):
        """
         A KNN classifier.
        :param strategy: The distance function to use. Must be an instance of Func Enum.
        :param k: The number of neighbors to consider.
        """

        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        self.y = None
        self.x = None
        self.strategy = strategies[strategy]
        self.k = k

    def fit(self, x, y):
        """
        Fit the model to the data.
        :param x: The training data.
        :param y: The training labels.
        :return:
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values

        self.x = x
        self.y = y

    def predict(self, x):
        if self.x is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")
        if isinstance(x, pd.DataFrame):
            x = x.values
        return np.array([self._predict(x_i) for x_i in x])

    def predict_single(self, x):
        if self.x is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")
        if len(x.shape) > 1:
            x = x.reshape(-1)
        if isinstance(x, pd.Series):
            x = x.values
        return self._predict(x)

    def _predict(self, point: np.array):
        distances = np.array([self.strategy(point, x_i) for x_i in self.x])
        indexes = np.argsort(distances)[:self.k]
        return Counter(self.y[indexes]).most_common(1)[0][0]

    def score(self, x: pd.DataFrame, y: pd.Series, *, metric: Metric = 'accuracy'):
        if self.x is None or self.y is None:
            raise ValueError("Model has not been fitted. Please call the `fit` method before scoring.")

        return metric_functions[metric](y, self.predict(x))

    def __repr__(self):
        return f'KNN(strategy={self.strategy.__name__}, k={self.k})'
