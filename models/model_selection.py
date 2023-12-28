from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models.metrics import global_confusion_matrix


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
