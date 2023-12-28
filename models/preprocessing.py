import numpy as np
import pandas as pd
from typing import Literal


def remove_rows_with_errors(input_df: pd.DataFrame) -> pd.DataFrame:
    error_rows = []

    for col in input_df.columns:
        try:
            input_df[col] = input_df[col].astype(float)
        except ValueError as e:
            print(f'could not convert data on column "{col}" with error {e}')
            error_rows.extend(input_df[col][pd.to_numeric(input_df[col], errors='coerce').isna()].index.tolist())

    error_rows = np.unique(error_rows)
    df_cleaned = input_df.drop(index=error_rows)
    print(f'removed rows are : {error_rows}')

    return df_cleaned


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


def replace_missing_values_with_mean(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.fillna(input_df.mean())


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


def treat_data(input_df: pd.DataFrame, *, normalization: Literal['minmax', 'z-score'] = 'z-score') -> pd.DataFrame:
    """
    Perform data treatment on a pandas DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): The input DataFrame.
    normalization (str): The normalization method to use. Must be one of 'minmax' or 'z-score'.

    Returns:
    pandas.DataFrame: DataFrame with values treated.
    """
    copy_df = input_df.copy()
    copy_df = remove_rows_with_errors(copy_df)
    copy_df = remove_duplicates_from_dataframe(copy_df)
    copy_df = replace_missing_values_with_mean(copy_df)

    if normalization == 'minmax':
        copy_df = min_max_normalization(copy_df)
    elif normalization == 'z-score':
        copy_df = z_score_normalization(copy_df)
    else:
        raise ValueError('Normalization must be one of "minmax" or "z-score".')

    return copy_df
