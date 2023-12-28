import numpy as np
from sklearn.utils import resample
from models.metrics import Metric, metric_functions
from models.tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=100, max_depth=150, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, x, y):
        for _ in range(self.n_trees):
            # Bootstrap sampling for creating a diverse set of trees
            bootstrap_x, bootstrap_y = resample(x, y, replace=True)

            # Create a decision tree and fit it to the bootstrap sample
            tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(bootstrap_x, bootstrap_y)
            self.trees.append(tree)

    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        return np.mean(predictions, axis=0) >= 0.5

    def score(self, x, y, *, metric: Metric = 'accuracy'):
        predictions = self.predict(x)
        return metric_functions[metric](y, predictions)

    def __repr__(self):
        return f'RandomForest(n_trees={self.n_trees}, max_depth={self.max_depth})'
