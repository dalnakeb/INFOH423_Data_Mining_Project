import pandas as pd
import ast
import numpy as np


def load_data_csv(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a dataframe.
    :param: file_path: The path to the CSV file.

    :return: pd.DataFrame: A DataFrame containing the csv data.
    """
    data_df = pd.read_csv(file_path)

    return data_df


def reformat_str_to_list(data_df: pd.DataFrame, cols: list[str], col_type: type) -> pd.DataFrame:
    """
    Given a dataframe with columns containing listed elements in string format, given a list of columns names
    and the associated type: transform string formated lists into numpy arrays.
    :param data_df:
    :param cols: columns names to be transformed
    :param col_type: columns desired type
    :return: dataframe with transformed string lists into numpy arrarys
    """
    def string_to_array(value):
        parsed_list = ast.literal_eval(value)
        return np.array(parsed_list).astype(col_type)

    for col_name in cols:
        data_df[col_name] = data_df[col_name].apply(lambda value: string_to_array(value))

    return data_df


def save_data(data_df: pd.DataFrame, filepath: str) -> None:
    """
    Save dataframe into CSV given a filepath. Transforms lists into strings for complete information retention
    :param data_df:
    :param filepath:
    :return: None
    """
    def np_to_list(row):
        for col in range(len(row)):
            if isinstance(row.iloc[col], np.ndarray):
                row.iloc[col] = str(list(row.iloc[col]))
        return row

    data_df = data_df.apply(np_to_list, axis=1)
    data_df.to_csv(filepath, index=False)