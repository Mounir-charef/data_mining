import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from models.utils import Func


class Kmeans:
    def __init__(self, k: int, *, func: Func = Func.euclidean, random_state=None, max_iter=1000):
        self.k = k
        self.func = func.value
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self._clusters_labels = None

    @property
    def clusters_labels(self):
        if self._clusters_labels is None:
            raise ValueError('You must assign labels first')
        return self._clusters_labels

    @property
    def labels(self):
        return self.clusters_labels[self.labels_]

    def _initialize_random_centroids(self, x: np.ndarray):
        centroids = np.zeros((self.k, x.shape[1]))

        for feature_idx in range(x.shape[1]):
            feature_min, feature_max = np.min(x[:, feature_idx]), np.max(x[:, feature_idx])
            centroids[:, feature_idx] = np.random.uniform(feature_min, feature_max, self.k)

        return centroids

    def _initialize_centroids(self, x: np.ndarray):
        np.random.seed(self.random_state)
        centroid_indices = np.random.choice(x.shape[0], self.k, replace=False)
        centroids = x[centroid_indices, :]
        return centroids

    def fit(self, x: np.ndarray):
        def is_converged(old_centroids, new_centroids):
            return np.array_equal(old_centroids, new_centroids)

        self.centroids = self._initialize_random_centroids(x) if not self.random_state else self._initialize_centroids(
            x)

        self.labels_ = np.zeros(x.shape[0], dtype=int)
        for _ in range(self.max_iter):
            old_cent = np.copy(self.centroids)
            distances = np.zeros((x.shape[0], self.k))
            for j in range(self.k):
                distances[:, j] = self.func(self.centroids[j], x)
            self.labels_ = np.argmin(distances, axis=1)
            for i in range(self.k):
                cluster_indices = np.where(self.labels_ == i)[0]
                self.centroids[i] = np.mean(x[cluster_indices], axis=0) if cluster_indices.size > 0 else self.centroids[
                    i]
            if is_converged(old_cent, self.centroids):
                break

    def predict(self, x):
        distances = np.zeros((x.shape[0], self.k))
        for j in range(self.k):
            distances[:, j] = self.func(self.centroids[j], x)
        clusters = np.argmin(distances, axis=1)
        return clusters

    def assign_labels(self, y):
        if self.centroids is None:
            raise ValueError('You must fit the model first')
        self._clusters_labels = np.zeros(self.k, dtype=int)
        for i in range(self.k):
            cluster_indices = np.where(self.labels_ == i)[0]
            cluster_labels = y[cluster_indices]
            if cluster_indices.size > 0:
                unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
                self._clusters_labels[i] = unique_labels[np.argmax(label_counts)]

    def silhouette_score(self, x):
        if self.centroids is None:
            raise ValueError('You must fit the model first')
        cluster_indices = [np.where(self.labels_ == i)[0] for i in range(self.k)]
        a = np.zeros(x.shape[0])
        b = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            cluster_idx = self.labels_[i]
            cluster = x[cluster_indices[cluster_idx]]
            a[i] = np.mean(self.func(x[i], cluster))
            b[i] = np.min([np.mean(self.func(x[i], x[cluster_indices[j]])) for j in range(self.k) if j != cluster_idx])
        return np.mean((b - a) / np.maximum(a, b))

    def score(self, y):
        if self.centroids is None:
            raise ValueError('You must fit the model first')
        accuracy = np.mean(y == self.labels) * 100
        return accuracy

    def plot(self, x):
        if self.centroids is None:
            raise ValueError('You must fit the model first')

        pca = PCA(n_components=2)
        reduced_x = pca.fit_transform(x)
        reduced_centroids = pca.transform(self.centroids)

        sns.scatterplot(x=reduced_x[:, 0], y=reduced_x[:, 1], hue=self.labels_, palette='bright')
        sns.scatterplot(x=reduced_centroids[:, 0], y=reduced_centroids[:, 1], color='black', marker='x', s=100)

        plt.show()
