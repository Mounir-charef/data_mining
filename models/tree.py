import numpy as np
from models.metrics import metric_functions, Metric
import pandas as pd


class DecisionTree:
    def __init__(self, max_depth=150):
        self.tree = None
        self.max_depth = max_depth

    def fit(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values
        self.tree = self._build_tree(x, y, 0)

    def _build_tree(self, x, y, depth):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            # If all labels are the same, create a leaf node or max depth reached or all labels are the same,
            # create a leaf node
            return {'label': y[0]}

        if x.shape[1] == 0:
            # If no features left, create a leaf node with the majority label
            return {'label': np.argmax(np.bincount(y))}

        # Find the best split
        best_feature, best_threshold = self._find_best_split(x, y)

        if best_feature is None:
            # Unable to find a split, create a leaf node with the majority label
            return {'label': np.argmax(np.bincount(y))}

        # Split the data based on the best feature and threshold
        left_mask = x[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build the left and right subtrees
        left_tree = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(x[right_mask], y[right_mask], depth + 1)

        # Return a node representing the split
        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left_tree, 'right': right_tree}

    def _find_best_split(self, x, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(x.shape[1]):
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                left_mask = x[:, feature] <= threshold
                right_mask = ~left_mask

                gini = (self._calculate_gini_impurity(y[left_mask])
                        * len(y[left_mask]) / len(y) + self._calculate_gini_impurity(y[right_mask]) * len(
                            y[right_mask]) / len(y))

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    @staticmethod
    def _calculate_gini_impurity(y):
        if len(y) == 0:
            return 0

        p_1 = np.sum(y) / len(y)
        p_0 = 1 - p_1

        gini = 1 - (p_0 ** 2 + p_1 ** 2)
        return gini

    def score(self, x, y, *, metric: Metric = 'accuracy'):
        predictions = self.predict(x)
        return metric_functions[metric](y, predictions)

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        predictions = []
        for row in x:
            predictions.append(self._predict_single(row, self.tree))
        return np.array(predictions, dtype=int)

    def predict_single(self, x):
        if isinstance(x, pd.Series):
            x = x.values

        if isinstance(x, list):
            x = np.array(x)

        if len(x.shape) > 1:
            x = x.reshape(-1)
        return int(self._predict_single(x, self.tree))

    def _predict_single(self, instance, node):
        if 'label' in node:
            # Leaf node, return the label
            return node['label']

        if instance[node['feature']] <= node['threshold']:
            # Recur on the left subtree
            return self._predict_single(instance, node['left'])
        else:
            # Recur on the right subtree
            return self._predict_single(instance, node['right'])

    def __repr__(self):
        return f'DecisionTree(max_depth={self.max_depth})'
