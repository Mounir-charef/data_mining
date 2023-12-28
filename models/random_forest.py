import numpy as np
from sklearn.utils import resample
from models.tree import DecisionTree
from tqdm import tqdm


class RandomForest:
    def __init__(self, n_trees=100, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, x, y):
        for _ in range(self.n_trees):
            # Bootstrap sampling for creating a diverse set of trees
            bootstrap_x, bootstrap_y = resample(x, y, replace=True)

            tree = DecisionTree()
            tree.fit(bootstrap_x, bootstrap_y)
            self.trees.append(tree)

    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in tqdm(self.trees)])
        return np.mean(predictions, axis=0) >= 0.5

    def score(self, x, y):
        predictions = self.predict(x)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy
