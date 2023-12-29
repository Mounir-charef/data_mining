import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from models.utils import strategies, Strategy
from models.metrics import metric_functions, Metric


class Kmeans:
    def __init__(self, n_clusters: int, *, func: Strategy = 'euclidean', random_state=None, max_iter=1000):
        self.n_clusters = n_clusters
        self.func = strategies[func]
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self._clusters_labels = None

    def _initialize_random_centroids(self, x: np.ndarray):
        centroids = np.zeros((self.n_clusters, x.shape[1]))

        for feature_idx in range(x.shape[1]):
            feature_min, feature_max = np.min(x[:, feature_idx]), np.max(x[:, feature_idx])
            centroids[:, feature_idx] = np.random.uniform(feature_min, feature_max, self.n_clusters)

        return centroids

    def _initialize_centroids(self, x: np.ndarray):
        np.random.seed(self.random_state)
        centroid_indices = np.random.choice(x.shape[0], self.n_clusters, replace=False)
        centroids = x[centroid_indices, :]
        return centroids

    def fit(self, x: np.ndarray, y=None):
        def is_converged(old, new_centroids):
            return np.array_equal(old, new_centroids)

        self.centroids = self._initialize_random_centroids(x) if not self.random_state else self._initialize_centroids(
            x)

        self.labels_ = np.zeros(x.shape[0], dtype=int)
        for _ in range(self.max_iter):
            old_centroids = np.copy(self.centroids)
            for i in range(x.shape[0]):
                self.labels_[i] = np.argmin([self.func(self.centroids[j], x[i]) for j in range(self.n_clusters)])
            for i in range(self.n_clusters):
                cluster_indices = np.where(self.labels_ == i)[0]
                self.centroids[i] = np.mean(x[cluster_indices], axis=0) if cluster_indices.size > 0 else self.centroids[
                    i]
            if is_converged(old_centroids, self.centroids):
                break

    def predict(self, x):
        clusters = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            clusters[i] = np.argmin([self.func(self.centroids[j], x[i]) for j in range(self.n_clusters)])
        return clusters

    def score(self, x_test, y_test, *, metric: Metric = 'accuracy'):
        if self.labels_ is None:
            raise Exception('You must fit the model first')

        return metric_functions[metric](y_test, self.predict(x_test))

    def plot(self, x):
        if self.centroids is None:
            raise Exception('You must fit the model first')

        pca = PCA(n_components=2)
        reduced_x = pca.fit_transform(x)
        reduced_centroids = pca.transform(self.centroids)

        sns.scatterplot(x=reduced_x[:, 0], y=reduced_x[:, 1], hue=self.labels_, palette='bright')
        sns.scatterplot(x=reduced_centroids[:, 0], y=reduced_centroids[:, 1], color='black', marker='x', s=100)

        plt.show()

    def __repr__(self):
        return f'Kmeans(n_clusters={self.n_clusters}, func={self.func.__name__},' \
               f' random_state={self.random_state}, max_iter={self.max_iter})'
