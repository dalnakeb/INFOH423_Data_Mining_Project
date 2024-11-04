import numpy as np
import pandas as pd


def get_rows_with_missing_data_points(data_df: pd.DataFrame) -> np.array:
    """
    looks for rows where the number of data points in the list type columns among all of these columns is different.
    :param data_df: dataframe
    :return: indices of rows with non-homogeneous list columns
    """
    list_columns = ['vehicles_sequence',
                    'events_sequence',
                    'seconds_to_incident_sequence',
                    'train_kph_sequence',
                    'dj_ac_state_sequence',
                    'dj_dc_state_sequence']

    equals = np.repeat(True, data_df.shape[0])
    for i in range(data_df.shape[0]):
        N = data_df.loc[i, list_columns[0]].shape[0]
        for col in list_columns:
            if data_df.loc[i, col].shape[0] != N:
                equals[i] = False

    return np.where(equals == False)


def get_rows_with_missing_data(data_df: pd.DataFrame) -> np.array:
    """
    Looks for rows that contain missing data in the list type columns (null data)
    :param: data_df: a data frame
    :return: the indices of rows containing missing data
    """
    list_columns = ['vehicles_sequence',
                    'events_sequence',
                    'seconds_to_incident_sequence',
                    'train_kph_sequence',
                    'dj_ac_state_sequence',
                    'dj_dc_state_sequence']

    equals = np.repeat(True, data_df.shape[0])
    for i in range(data_df.shape[0]):
        for col in list_columns:
            if np.isnan(data_df.loc[i, col]).any():
                equals[i] = False

    return np.where(equals == False)
