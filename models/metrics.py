import numpy as np
from typing import Literal

from models.utils import Strategy, strategies

Average = Literal['macro', 'micro', 'weighted', 'binary']


def metrics_by_class(y_true, y_pred, classes=None):
    metrics = {}
    matrices = {}
    if classes is None:
        classes = np.unique(y_true)
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        tn = np.sum((y_true != c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        matrices[c] = [[tp, fp], [fn, tn]]
        metrics[c] = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0,
            'precision': tp / (tp + fp) if tp + fp > 0 else 0,
            'recall': tp / (tp + fn) if tp + fn > 0 else 0,
            'f1-score': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0,
            'specificity': tn / (tn + fp) if tn + fp > 0 else 0
        }
    return metrics, matrices


def accuracy_score(y_true, y_pred, **kwargs):
    return np.sum(y_true == y_pred) / len(y_true)


def precision_score(y_true, y_pred, *, average: Average = 'macro'):
    classes = np.unique(y_true)
    metrics, matrices = metrics_by_class(y_true, y_pred, classes)

    match average:
        case 'macro':
            return np.mean([metrics[c]['precision'] for c in classes])
        case 'micro':
            matrix = np.sum([matrices[c] for c in classes], axis=0)
            tp = matrix[0][0]
            fp = matrix[0][1]
            return tp / (tp + fp)
        case 'weighted':
            return np.sum([metrics[c]['precision'] * np.sum(y_true == c) / len(y_true) for c in classes])
        case 'binary':
            if len(classes) != 2:
                raise ValueError('Binary average requires exactly 2 classes')
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp)
        case _:
            raise ValueError(f'Invalid average {average!r}')


def recall_score(y_true, y_pred, *, average: Average = 'macro'):
    classes = np.unique(y_true)
    metrics, matrices = metrics_by_class(y_true, y_pred, classes)

    match average:
        case 'macro':
            return np.mean([metrics[c]['recall'] for c in classes])
        case 'micro':
            matrix = np.sum([matrices[c] for c in classes], axis=0)
            tp = matrix[0][0]
            fn = matrix[1][0]
            return tp / (tp + fn)
        case 'weighted':
            return np.sum([metrics[c]['recall'] * np.sum(y_true == c) / len(y_true) for c in classes])
        case 'binary':
            if len(classes) != 2:
                raise ValueError('Binary average requires exactly 2 classes')
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn)
        case _:
            raise ValueError(f'Invalid average {average!r}')


def f1_score(y_true, y_pred, *, average: Average = 'macro'):
    classes = np.unique(y_true)
    metrics, matrices = metrics_by_class(y_true, y_pred, classes)

    match average:
        case 'macro':
            return np.mean([metrics[c]['f1-score'] for c in classes])
        case 'micro':
            matrix = np.sum([matrices[c] for c in classes], axis=0)
            tp = matrix[0][0]
            fp = matrix[0][1]
            fn = matrix[1][0]
            return 2 * tp / (2 * tp + fp + fn)
        case 'weighted':
            return np.sum([metrics[c]['f1-score'] * np.sum(y_true == c) / len(y_true) for c in classes])
        case 'binary':
            if len(classes) != 2:
                raise ValueError('Binary average requires exactly 2 classes')
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return 2 * tp / (2 * tp + fp + fn)
        case _:
            raise ValueError(f'Invalid average {average!r}')


def specificity_score(y_true, y_pred, *, average: Average = 'macro'):
    classes = np.unique(y_true)
    metrics, matrices = metrics_by_class(y_true, y_pred, classes)

    match average:
        case 'macro':
            return np.mean([metrics[c]['specificity'] for c in classes])
        case 'micro':
            matrix = np.sum([matrices[c] for c in classes], axis=0)
            tn = matrix[1][1]
            fp = matrix[0][1]
            return tn / (tn + fp)
        case 'weighted':
            return np.sum([metrics[c]['specificity'] * np.sum(y_true == c) / len(y_true) for c in classes])
        case 'binary':
            if len(classes) != 2:
                raise ValueError('Binary average requires exactly 2 classes')
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tn / (tn + fp)
        case _:
            raise ValueError(f'Invalid average {average!r}')


def global_confusion_matrix(y_true, y_pred, classes=None):
    if classes is None:
        classes = np.unique(y_true)

    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            matrix[i][j] = np.sum((y_true == c1) & (y_pred == c2))
    return matrix


def print_confusion_matrix(y_true, y_pred, classes=None):
    if classes is None:
        classes = np.unique(y_true)

    matrix = global_confusion_matrix(y_true, y_pred, classes)
    print('Confusion matrix:')
    print('  ' + ' '.join([f'{c:^4}' for c in classes]))
    for i, c1 in enumerate(classes):
        print(f'{c1} ' + ' '.join([f'{matrix[i][j]:^4}' for j, c2 in enumerate(classes)]))


def silhouette_score(x, y_pred, *, strategy: Strategy = 'euclidean'):
    """
    Compute the silhouette score for a clustering.
    :param x:
    :param y_pred:
    :param strategy:
    :return:
    """
    func = strategies[strategy]
    clusters = np.unique(y_pred)
    if len(clusters) == 1:
        return np.nan
    score = 0
    for i in range(len(x)):
        a_i = np.mean([func(x[i], x[j]) for j in range(len(x)) if y_pred[j] == y_pred[i] and i != j])
        b_i = np.min([np.mean([func(x[i], x[j]) for j in range(len(x)) if y_pred[j] == k])
                      for k in clusters if k != y_pred[i]])
        score += (b_i - a_i) / max(a_i, b_i)
    return score / len(x)


metric_functions = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1-score': f1_score,
    'specificity': specificity_score
}

Metric = Literal['accuracy', 'precision', 'recall', 'f1-score', 'specificity']
