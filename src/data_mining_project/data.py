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


def reformat_str_to_list(data_df, cols: list[str], col_type: type):
    def string_to_array(value):
        parsed_list = ast.literal_eval(value)
        return np.array(parsed_list).astype(col_type)

    for col_name in cols:
        data_df[col_name] = data_df[col_name].apply(lambda value: string_to_array(value))

    return data_df


def save_data(data_df, filepath):
    def np_to_list(row):
        for col in range(len(row)):
            if isinstance(row.iloc[col], np.ndarray):
                row.iloc[col] = str(list(row.iloc[col]))
        return row

    data_df = data_df.apply(np_to_list, axis=1)
    data_df.to_csv(filepath, index=False)