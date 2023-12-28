from models.utils import Strategy, strategies
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


class KNN:
    def __init__(self, k: int = 2, *, strategy: Strategy = strategies['euclidean']):
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

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """
        Fit the model to the data.
        :param x: The training data.
        :param y: The training labels.
        :return:
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        self.x = x
        self.y = y

    def predict(self, x: pd.DataFrame):
        if self.x is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")

        return np.array([self._predict(x_i) for x_i in x.values])

    def _predict(self, point: np.array):
        distances = np.array([self.strategy(point, x_i) for x_i in self.x.values])
        indexes = np.argsort(distances)[:self.k]
        return Counter(self.y.values[indexes]).most_common(1)[0][0]

    def score(self, x: pd.DataFrame, y: pd.Series):
        if self.x is None or self.y is None:
            raise ValueError("Model has not been fitted. Please call the `fit` method before scoring.")

        return accuracy_score(y, self.predict(x))

    def __repr__(self):
        return f'KNN(strategy={self.strategy.__name__}, k={self.k})'
