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


def reformat_data(data_df):
    list_columns_int = [
        'vehicles_sequence',
        'events_sequence',
        'seconds_to_incident_sequence',
        'dj_ac_state_sequence',
        'dj_dc_state_sequence',
    ]

    list_columns_float = [
        'train_kph_sequence'
    ]

    def string_to_array_float(value):
        parsed_list = ast.literal_eval(value)
        return np.array(parsed_list).astype(float)

    def string_to_array_int(value):
        parsed_list = ast.literal_eval(value)
        return np.array(parsed_list).astype(int)

    for col in list_columns_int:
        data_df[col] = data_df[col].apply(string_to_array_int)

    for col in list_columns_float:
        data_df[col] = data_df[col].apply(string_to_array_float)
    return data_df
