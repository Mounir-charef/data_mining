import numpy as np
import pandas as pd
from typing import Literal
from collections import Counter

from pandas import DataFrame


def custom_describe(input_df: pd.DataFrame):
    result = {}

    for column in input_df.columns:
        sorted_values = sorted(input_df[column].tolist())
        # Maximum
        max_val = sorted_values[-1]

        # Minimum
        min_val = sorted_values[0]

        # Mean
        mean = sum(sorted_values) / len(sorted_values)

        # Mode
        counter = Counter(input_df[column])
        mode = counter.most_common(1)[0][0]

        # Median
        n = len(sorted_values)
        if n % 2 == 0:
            median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            median = sorted_values[n // 2]

        # Standard Deviation
        std_val = (sum((x - mean) ** 2 for x in sorted_values) / len(sorted_values)) ** 0.5

        # Quantiles
        quantiles = {
            'Q0': min_val,
            'Q1': sorted_values[int(0.25 * n)],
            'Q2': median,
            'Q3': sorted_values[int(0.75 * n)],
            'Q4': max_val
        }

        result[column] = {
            'max': max_val,
            'min': min_val,
            'mean': mean,
            'mode': mode,
            'median': median,
            'std': std_val,
            'quantiles': quantiles
        }

    return result


def remove_duplicates_from_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    seen_rows = set()
    output_rows = []

    for index, row in input_df.iterrows():
        row_tuple = tuple(row)
        if row_tuple not in seen_rows:
            seen_rows.add(row_tuple)
            output_rows.append(row)

    output_df = pd.DataFrame(output_rows, columns=input_df.columns)
    return output_df


def treat_rows_with_missing_values(input_df: pd.DataFrame, *, values=None) -> pd.DataFrame:
    if values is None:
        values = [
            "",
            " ",
            "nan",
            "NaN",
            "Nan",
            "NAN",
            "None",
            "none",
            "NONE",
            "null",
            "Null",
            "NULL",
            "?",
            "NA",
            "na",
            "Na",
            "nA",
            None,
            np.nan,
        ]
    output_df = input_df.copy()
    output_df[output_df.isin(values)] = np.nan
    output_df = output_df.astype(float)
    column_means = output_df.mean()
    output_df = output_df.fillna(pd.Series(column_means))
    return output_df


def z_score_normalization(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform z-score normalization on each column of a pandas DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: DataFrame with values normalized using z-score normalization.
    """
    copy_df = input_df.copy()
    means = copy_df.mean()
    stds = copy_df.std()
    copy_df = (copy_df - means) / stds

    return copy_df


def min_max_normalization(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform min-max normalization on each column of a pandas DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: DataFrame with values normalized using min-max normalization.
    """
    copy_df = input_df.copy()
    min_ = copy_df.min()
    max_ = copy_df.max()
    copy_df = (copy_df - min_) / (max_ - min_)

    return copy_df


def has_outliers(input_df: pd.DataFrame, *, threshold: float = 1.5) -> bool:
    data_description = custom_describe(input_df)
    for column in input_df.columns:
        values = input_df[column]
        q1 = data_description[column]['quantiles']['Q1']
        q3 = data_description[column]['quantiles']['Q3']
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        column_outliers = values[(values < lower_bound) | (values > upper_bound)].index.to_list()

        if column_outliers:
            return True

    return False


def detect_and_treat_outliers(dataframe: pd.DataFrame, *, threshold: float = 1.5, show: bool = False) -> pd.DataFrame:
    data_description = custom_describe(dataframe)
    for column in dataframe.columns:
        values = dataframe[column]
        q1 = data_description[column]['quantiles']['Q1']
        q3 = data_description[column]['quantiles']['Q3']
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        column_outliers = values[(values < lower_bound) | (values > upper_bound)].index.to_list()

        if show and column_outliers:
            print(f"Outliers in column '{column}': {column_outliers}")

        values[column_outliers] = values.mean()
        dataframe[column] = values
    return dataframe


def treat_outliers(input_df: pd.DataFrame, show: bool = False) -> pd.DataFrame:
    while True:
        df_without_outliers = detect_and_treat_outliers(input_df, show=show)
        if not has_outliers(df_without_outliers):
            return df_without_outliers
        input_df = df_without_outliers


def treat_data(input_df: pd.DataFrame, *, target_column: str = 'Fertility',
               normalization: Literal['minmax', 'z-score'] = 'z-score') -> \
        tuple[DataFrame, pd.Series]:
    """
    Perform data treatment on a pandas DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): The input DataFrame.
    normalization (str): The normalization method to use. Must be one of 'minmax' or 'z-score'.

    Returns:
    pandas.DataFrame: DataFrame with values treated.
    """
    copy_df = input_df.copy()
    copy_df = remove_duplicates_from_dataframe(copy_df)
    copy_df = treat_rows_with_missing_values(copy_df)
    copy_df = treat_outliers(copy_df)
    print(copy_df.describe())

    y = copy_df[target_column]
    x = copy_df.drop(columns=target_column)

    if normalization == 'minmax':
        x = min_max_normalization(x)
    elif normalization == 'z-score':
        x = z_score_normalization(x)
    else:
        raise ValueError('Normalization must be one of "minmax" or "z-score".')

    return x, y
