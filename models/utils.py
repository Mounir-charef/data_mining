import numpy as np
from typing import Literal


def euclidean_distance(x, y, axis=None):
    return np.linalg.norm(x - y, axis=axis)


def manhattan_distance(x, y, axis=None):
    return np.sum(np.abs(x - y), axis=axis)


def minkowski_distance(x, y, p=2, axis=None):
    return np.sum(np.abs(x - y) ** p, axis=axis) ** (1 / p)


def cosine_distance(vector1, vector2):
    return 1 - (np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


def hamming_distance(x, y, axis=None):
    return np.sum(x != y, axis=axis)


Strategy = Literal['euclidean', 'manhattan', 'minkowski', 'cosine', 'hamming']

strategies = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'minkowski': minkowski_distance,
    'cosine': cosine_distance,
    'hamming': hamming_distance
}
