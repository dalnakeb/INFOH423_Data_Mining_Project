import pandas as pd
import ast
import numpy as np


def load_data_csv(file_path):
    """
    Load data from a CSV file into a dataframe.
    :param: file_path (str): The path to the CSV file.

    :return: pd.DataFrame: A DataFrame containing the csv data.
    """
    data_df = pd.read_csv(file_path)
    list_columns = [
        'vehicles_sequence',
        'events_sequence',
        'seconds_to_incident_sequence',
        'train_kph_sequence',
        'dj_ac_state_sequence',
        'dj_dc_state_sequence'
    ]

    def string_to_array(value):
        parsed_list = ast.literal_eval(value)
        return np.array(parsed_list, dtype=float)

    for col in list_columns:
        data_df[col] = data_df[col].apply(string_to_array)

    return data_df


