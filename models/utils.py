import numpy as np
from typing import Literal


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def minkowski_distance(x, y, p=2):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def cosine_distance(vector1, vector2):
    return 1 - (np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


def hamming_distance(x, y):
    return np.sum(x != y)


Strategy = Literal['euclidean', 'manhattan', 'minkowski', 'cosine', 'hamming']

strategies = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'minkowski': minkowski_distance,
    'cosine': cosine_distance,
    'hamming': hamming_distance
}
