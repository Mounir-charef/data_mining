import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models.utils import strategies, Strategy


class DBScan:
    def __init__(self, min_samples: int = 4, eps: float = 0.5, strategy: Strategy = "euclidean"):
        self.cluster_per_points = None
        self.min_samples = min_samples
        self.eps = eps
        self.strategy = strategies[strategy]

    def _calculate_distance(self, x, point_index, y_index):
        return self.strategy(x[point_index], x[y_index])

    def _get_neighbors(self, point_index: int, x: np.ndarray):
        neighbors_index = []
        for y_index, point_y in enumerate(x):
            distance = self._calculate_distance(x, point_index, y_index)
            if y_index != point_index and distance <= self.eps:
                neighbors_index.append(y_index)
        return neighbors_index

    def _is_core(self, point_index: int, x: np.array):
        return len(self._get_neighbors(point_index, x)) >= self.min_samples

    def _visit_neighbors(self, point_index: int, x: np.array, cluster_index: int):
        for neighbor_index in self._get_neighbors(point_index, x):
            if self.cluster_per_points[neighbor_index] == -1:
                self.cluster_per_points[neighbor_index] = cluster_index
                self._visit_neighbors(neighbor_index, x, cluster_index)

    def fit(self, x: pd.DataFrame):
        x = x.values
        self.cluster_per_points = np.full(shape=x.shape[0], fill_value=-1)
        cluster_index = 0
        for point_index, point in enumerate(x):
            if self.cluster_per_points[point_index] != -1:
                continue
            if self._is_core(point_index, x):
                self.cluster_per_points[point_index] = cluster_index
                self._visit_neighbors(point_index, x, cluster_index)
                cluster_index += 1

    def plot(self, x):
        plt.figure(figsize=(8, 6))
        if self.cluster_per_points is None:
            raise Exception('You must fit the model first')
        pca = PCA(n_components=2)
        reduced_x = pca.fit_transform(x)

        for label in np.unique(self.cluster_per_points):
            if label == -1:
                outliers_indices = np.where(np.array(self.cluster_per_points) == label)
                plt.scatter(reduced_x[outliers_indices, 0], reduced_x[outliers_indices, 1], c='black', marker='x',
                            label=f'Cluster {label} (Outliers)')
            else:
                cluster_indices = np.where(np.array(self.cluster_per_points) == label)
                plt.scatter(reduced_x[cluster_indices, 0], reduced_x[cluster_indices, 1], label=f'Cluster {label}')

        plt.legend()
        plt.show()
