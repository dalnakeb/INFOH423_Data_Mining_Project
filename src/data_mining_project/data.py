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
