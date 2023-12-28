from enum import Enum
from itertools import product
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models.metrics import global_confusion_matrix, metric_functions, metrics_by_class


class PlotType(Enum):
    BOX = 'Box Plot'
    HIST = 'Histogram'
    SCATTER = 'Scatter Plots'


def plot(input_df, *, plot_type: PlotType) -> None:
    num_cols = len(input_df.columns)
    num_rows = (num_cols + 1) // 7
    fig, axes = plt.subplots(num_rows, 7, figsize=(15, num_rows * 4))
    fig.suptitle(f'{plot_type.value} of Data', y=1.02)

    axes = axes.flatten()

    for i, column in enumerate(input_df.columns):
        match plot_type:
            case PlotType.BOX:
                sns.boxplot(y=input_df[column], ax=axes[i])
            case PlotType.HIST:
                sns.histplot(input_df[column], ax=axes[i], kde=True)
            case PlotType.SCATTER:
                sns.scatterplot(input_df[column], ax=axes[i])
        axes[i].set_title(column)

    plt.tight_layout()
    plt.show()


def heatmaps(x_train, y_train, x_test, y_test, models):
    """
        plot the heatmaps of the models in a grid
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param models:
    :return:
    """
    num_rows = (len(models) + 1) // 2
    fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
    fig.suptitle('Heatmaps of the models')

    for i, model in enumerate(models):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        matrix = global_confusion_matrix(y_test, y_pred)
        # plot the heatmap
        sns.heatmap(matrix, annot=True, fmt='d', ax=axs[i // 2][i % 2])
        axs[i // 2][i % 2].set_title(str(model))
        axs[i // 2][i % 2].set_xlabel('Predicted')
        axs[i // 2][i % 2].set_ylabel('Actual')
    # remove the extra plots
    for i in range(len(models), num_rows * 2):
        axs[i // 2][i % 2].remove()
    plt.tight_layout()
    plt.show()


def train_test_split(x: pd.DataFrame, y: pd.Series, test_size=0.2):
    """
        split the data into train and test sets based on the test size after shuffling
    :param x:
    :param y:
    :param test_size:
    :return:
    """
    # shuffle the data
    shuffled = pd.concat([x, y], axis=1).sample(frac=1)
    # split the data
    split_index = round(len(shuffled) * test_size)
    x_train = shuffled.iloc[split_index:, :-1]
    y_train = shuffled.iloc[split_index:, -1]
    x_test = shuffled.iloc[:split_index, :-1]
    y_test = shuffled.iloc[:split_index, -1]
    return x_train, x_test, y_train, y_test


def evaluate(x_train, y_train, x_test, y_test, models: list, metrics: Iterable, averages: Iterable):
    """
        evaluate the models on the given metrics
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param models:
    :param metrics:
    :param averages:
    :return:
    """
    results = []
    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        result = {'model': str(model)}
        for metric in metrics:
            if metric == 'accuracy':
                result[metric] = metric_functions[metric](y_test, y_pred)
            else:
                for average in averages:
                    result[f'{metric}_({average})'] = metric_functions[metric](y_test, y_pred, average=average)
        results.append(result)
    return pd.DataFrame(results)


def evaluate_by_class(x_train, y_train, x_test, y_test, models: list):
    """
        evaluate the models on the given metrics by class
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param models:
    :return:
    """
    results = {}
    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics, _ = metrics_by_class(y_test, y_pred)
        results[str(model)] = metrics
    return results


def grid_search(model_cls, x_train: pd.DataFrame, y_train: pd.Series,
                x_test: pd.DataFrame, y_test: pd.Series, param_grid: dict, metrics: Iterable, averages: Iterable):
    """
    Perform grid search for hyperparameter tuning and evaluate on specified metrics.

    Parameters:
    - model_cls: The class of the model to be tuned.
    - x_train: Training features.
    - y_train: Training labels.
    - x_test: Testing features.
    - y_test: Testing labels.
    - param_grid: Dictionary of hyperparameter values to search over.
    - metrics: Iterable of metrics to compute.
    - averages: Iterable of averages for applicable metrics.

    Returns:
    - Dictionary containing evaluation results for the best hyperparameter combination.
    - Dictionary containing the best hyperparameter combination.
    """
    scores = []
    best_results = {'score': float('-inf')}
    # Iterate through all combinations of hyperparameter values
    for params in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), params))
        # Create an instance of the model class with the current hyperparameters
        model = model_cls(**params)
        # Train the model
        model.fit(x_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(x_test)
        # Evaluate performance on specified metrics
        score = 0
        results = {'model': str(model)}
        for metric in metrics:
            if metric == 'accuracy':
                metric_score = metric_functions[metric](y_test, y_pred)
                results[metric] = metric_score
                score += metric_score
            else:
                for average in averages:
                    metric_average = metric_functions[metric](y_test, y_pred, average=average)
                    results[f'{metric}_({average})'] = metric_average
                    score += metric_average

        # Update the best hyperparameters if the current model performs better
        print(f'{model}: {score}')
        results['score'] = score
        if score > best_results['score']:
            best_results = {
                'model': str(model),
                'score': score,
                'params': params
            }
        scores.append(results)

    return pd.DataFrame(scores), best_results
