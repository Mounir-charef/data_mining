from enum import Enum
from itertools import product
from typing import Iterable, get_args
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm
from models import KMeans
from models.metrics import global_confusion_matrix, metric_functions, metrics_by_class, Metric, Average, \
    silhouette_score
from models.utils import Strategy


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


def confusion_matrices(x_test, y_test, models):
    """
        plot the heatmaps of the models in a grid
    :param x_test:
    :param y_test:
    :param models:
    :return:
    """
    num_rows = (len(models) + 1) // 2
    fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
    fig.suptitle('Heatmaps of the models')

    for i, model in enumerate(models):
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


def evaluate(x_test, y_test, models: list, metrics: Iterable[Metric], averages: Iterable[Average]):
    """
        evaluate the models on the given metrics
    :param x_test:
    :param y_test:
    :param models:
    :param metrics:
    :param averages:
    :return:
    """
    results = []
    for model in models:
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


def evaluate_by_class(x_test, y_test, models: list):
    """
        evaluate the models on the given metrics by class
    :param x_test:
    :param y_test:
    :param models:
    :return:
    """
    results = {}
    for model in models:
        y_pred = model.predict(x_test)
        metrics, _ = metrics_by_class(y_test, y_pred)
        results[str(model)] = metrics
    return results


def grid_search_cv(model_cls, x_train: pd.DataFrame, y_train: pd.Series, param_grid: dict,
                   metric: Metric, average: Average, cv: int = 5):
    """
    Perform grid search for hyperparameter tuning and evaluate on specified metrics.

    Parameters:
    - model_cls: The class of the model to be tuned.
    - x_train: Training features.
    - y_train: Training labels.
    - param_grid: Dictionary of hyperparameter values to search over.
    - metric: Metric to evaluate on.
    - average: Average to use for multiclass classification.
    - cv: Number of folds for cross-validation.

    Returns:
    - A DataFrame of scores for each combination of hyperparameter values.
    - A dictionary of the best hyperparameters.
    """

    def create_folds(data, cv_):
        """
        Create folds for cross-validation.

        Parameters:
        - data: Data to be split into folds.
        - cv_: Number of folds.

        Returns:
        - A list of folds.
        """
        # Shuffle data
        shuffled = data.sample(frac=1)
        # Split data into folds
        fold_size = len(data) // cv_
        folds_ = []
        for k in range(cv_):
            start = k * fold_size
            end = (k + 1) * fold_size
            folds_.append(shuffled.iloc[start:end])
        return folds_

    # Get all combinations of hyperparameter values
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combos = list(product(*param_values))

    # Initialize DataFrame to store scores

    # Perform grid search
    best_score = 0
    best_params = {}
    scores = []
    for combo in param_combos:
        # Create model with hyperparameter values
        params = dict(zip(param_names, combo))
        model = model_cls(**params)
        # Create folds for cross-validation
        folds = create_folds(pd.concat([x_train, y_train], axis=1), cv)

        results = {'model': str(model)}
        # Compute scores for each fold
        fold_scores = []
        for i in range(cv):
            # Split data into train and validation sets
            train = pd.concat(folds[:i] + folds[i + 1:])
            valid = folds[i]
            x_train_ = train.iloc[:, :-1]
            y_train_ = train.iloc[:, -1]
            x_valid = valid.iloc[:, :-1]
            y_valid = valid.iloc[:, -1]

            # Fit model and predict
            model.fit(x_train_, y_train_)
            y_pred = model.predict(x_valid)
            fold_score = {}
            for metric_ in get_args(Metric):
                fold_score[metric_] = metric_functions[metric_](y_valid, y_pred, average=average)
            fold_scores.append(fold_score)

        # Compute average score across folds
        for metric_ in get_args(Metric):
            results[metric_] = sum([score[metric_] for score in fold_scores]) / cv
        scores.append(results)
        # Update best score and parameters depending on the selected metric
        if results[metric] > best_score:
            best_score = results[metric]
            best_params = params

    return pd.DataFrame(scores), {'best_score': best_score, 'best_params': best_params}


def plot_silhouette_scores(x_train, k_range: range, *, strategy: Strategy = 'euclidean'):
    """
        plot the elbow method for kmeans
    :param x_train:
    :param k_range:
    :param strategy:
    :return:
    """
    scores = []
    for k in tqdm(k_range):
        model = KMeans(k, random_state=42, distance_metric=strategy)
        model.fit(x_train)
        scores.append(silhouette_score(x_train, model.labels_, strategy=strategy))
    plt.plot(k_range, scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette scores for different k values')
    plt.show()


def plot_elbow_method(x_train, k_range: range, *, strategy: Strategy = 'euclidean'):
    """
        plot the elbow method for kmeans
    :param x_train:
    :param k_range:
    :param strategy:
    :return:
    """
    scores = []
    for k in tqdm(k_range):
        model = KMeans(k, random_state=42, distance_metric=strategy)
        model.fit(x_train)
        scores.append(model.inertia_)
    plt.plot(k_range, scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow method for different k values')
    plt.show()


def plot_cluster_by_model(x, y, title, input_models):
    """
        Plot in a grid the clusters of the data by each model after applying PCA
        to reduce the dimensionality to 2D

        :param x: data to be clustered
        :param y: actual labels
        :param title: title of the plot
        :param input_models: list of models to be used for clustering, models must be fitted
    """
    n_models = len(input_models)
    n_rows = 2
    n_cols = (n_models + 2) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)  # Larger title font size

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)

    for i, model in enumerate(input_models):
        row_idx = i // n_cols
        col_idx = i % n_cols
        y_pred = model.predict(x)

        sns.scatterplot(
            ax=axes[row_idx, col_idx],
            x=x_pca[:, 0],
            y=x_pca[:, 1],
            hue=y_pred,
            palette='Set2',
            s=80,  # Adjusted marker size
            alpha=0.8,  # Partial transparency for better overlap visualization
        )

        axes[row_idx, col_idx].set_title(model.__class__.__name__, fontsize=12)
        axes[row_idx, col_idx].legend(fontsize=10)

    # Plot actual labels
    sns.scatterplot(
        ax=axes[-1, -1],
        x=x_pca[:, 0],
        y=x_pca[:, 1],
        hue=y,
        palette='Set2',
        s=80,
        alpha=0.8,
    )

    axes[-1, -1].set_title('Actual', fontsize=12)
    axes[-1, -1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()
