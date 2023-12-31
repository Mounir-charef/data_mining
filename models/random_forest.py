import numpy as np
from sklearn.utils import resample
from models.metrics import Metric, metric_functions
from models.tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=100, max_depth=150):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        for _ in range(self.n_trees):
            bootstrap_x, bootstrap_y = resample(x, y, replace=True)

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(bootstrap_x, bootstrap_y)
            self.trees.append(tree)

    def predict(self, x):
        x = np.asarray(x)
        predictions = np.array([tree.predict(x) for tree in self.trees])

        majority_votes = np.apply_along_axis(
            lambda current: np.bincount(current).argmax(), axis=0, arr=predictions
        )

        return majority_votes

    def predict_single(self, x):
        x = np.asarray(x)
        predictions = np.array([tree.predict_single(x) for tree in self.trees])

        majority_votes = np.apply_along_axis(
            lambda current: np.bincount(current).argmax(), axis=0, arr=predictions
        )

        return majority_votes.item()

    def score(self, x, y, *, metric: Metric = 'accuracy'):
        predictions = self.predict(x)
        return metric_functions[metric](y, predictions)

    def __repr__(self):
        return f'RandomForest(n_trees={self.n_trees}, max_depth={self.max_depth})'
