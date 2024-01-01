import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

from models.metrics import Metric, metric_functions
from models.utils import Strategy, strategies


class KMeans:
    def __init__(self, num_clusters: int,
                 distance_metric: Strategy = 'euclidean',
                 max_iter: int = 300, random_state: int = None):
        """
        KMeans clustering algorithm.

        Parameters:
        - num_clusters (int): Number of clusters.
        - distance_metric (Literal): Distance metric strategy.
        - max_iter (int): Maximum number of iterations.
        - random_state (int): Seed for random initialization of centroids.
        """
        self.num_clusters = num_clusters
        self.distance_metric = strategies[distance_metric]
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self._labels = None
        self._inertia = None

    @property
    def labels_(self):
        if self._labels is None:
            raise Exception('You must fit the model first')
        return self._labels

    @property
    def inertia_(self):
        if self._inertia is None:
            raise Exception('You must fit the model first')
        return self._inertia

    def _initialize_random_centroids(self, x: np.ndarray):
        centroids = np.zeros((self.num_clusters, x.shape[1]))

        for feature_idx in range(x.shape[1]):
            feature_min, feature_max = np.min(x[:, feature_idx]), np.max(x[:, feature_idx])
            centroids[:, feature_idx] = np.random.uniform(feature_min, feature_max, self.num_clusters)

        return centroids

    def _initialize_centroids(self, x: np.ndarray):
        np.random.seed(self.random_state)
        centroid_indices = np.random.choice(x.shape[0], self.num_clusters, replace=False)
        centroids = x[centroid_indices, :]
        return centroids

    def _assign_labels(self, x: np.ndarray):
        for point in range(x.shape[0]):
            distances = np.array([self.distance_metric(x[point], centroid) for centroid in self.centroids])
            self._labels[point] = np.argmin(distances)

    def _update_centroids(self, x: np.ndarray):
        for i in range(self.num_clusters):
            self.centroids[i] = np.mean(x[self._labels == i], axis=0)

    def fit(self, x: pd.DataFrame, y=None):
        """
        Fit the KMeans model to the input data.

        Parameters:
        - x (pd.DataFrame): Input data.
        """
        x = np.array(x)
        self.centroids = self._initialize_centroids(x)
        self._labels = np.zeros(x.shape[0], dtype=int)
        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            self._assign_labels(x)
            self._update_centroids(x)
            if np.allclose(old_centroids, self.centroids):
                break

        self._inertia = 0
        for point in range(x.shape[0]):
            distances = np.array([self.distance_metric(x[point], centroid) for centroid in self.centroids])
            self._inertia += np.min(distances)

    def _predict(self, point: np.ndarray):
        """
        Predict the cluster label for a single data point.

        Parameters:
        - point (np.ndarray): Data point.

        Returns:
        """
        distances = np.array([self.distance_metric(point, centroid) for centroid in self.centroids])
        return np.argmin(distances)

    def predict(self, x: pd.DataFrame):
        """
        Predict the cluster labels for the input data.
        :param x:
        :return:
        """
        if self.centroids is None:
            raise Exception('You must fit the model first')

        return np.array([self._predict(x_i) for x_i in x.values])

    def predict_single(self, x: pd.Series):
        """
        Predict the cluster label for a single data point.

        Parameters:
        - x (pd.Series): Data point.

        Returns:
        """
        if self.centroids is None:
            raise Exception('You must fit the model first')

        return self._predict(x.values)

    def score(self, x: pd.DataFrame, y: pd.DataFrame, metric: Metric = 'accuracy'):
        """
        Calculate the inertia (sum of squared distances to the nearest centroid) as a score.

        Parameters:
        - x (pd.DataFrame): Input data.

        Returns:
        - float: Inertia score.
        """
        predictions = self.predict(x)
        return metric_functions[metric](y, predictions)

    def plot(self, x):
        if self.centroids is None:
            raise Exception('You must fit the model first')

        pca = PCA(n_components=2)
        reduced_x = pca.fit_transform(x)
        reduced_centroids = pca.transform(self.centroids)

        sns.scatterplot(x=reduced_x[:, 0], y=reduced_x[:, 1], hue=self._labels, palette='bright')
        sns.scatterplot(x=reduced_centroids[:, 0], y=reduced_centroids[:, 1], color='black', marker='x', s=100)

        plt.show()

    def __repr__(self):
        return f'KMeans(num_clusters={self.num_clusters}, ' \
               f'distance_metric={self.distance_metric.__name__}, max_iter={self.max_iter})'
