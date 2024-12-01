import numpy as np


def compute_events_freq(events_sequence):
    unique_events, events_count = np.unique(events_sequence, return_counts=True)
    total_events_count = np.sum(events_count)
    events_freq = events_count / total_events_count
    events_freq = np.column_stack((unique_events, events_freq))
    events_freq = {int(k): v for k, v in events_freq}
    return events_freq


def compute_events_to_remove(data_df, events_freq_all_classes, t):
    events_to_remove = {}
    events_freq_per_incident = {}
    for incident_type, events_sequences in list(data_df.groupby("incident_type")["events_sequence"]):
        events_sequence_per_incident = np.concatenate(events_sequences.to_list())
        events_freq_per_incident[incident_type] = compute_events_freq(events_sequence_per_incident)

    for incident_type, events_freq in events_freq_per_incident.items():
        events_to_remove[incident_type] = []
        for event_id, event_freq in events_freq.items():
            if event_freq / events_freq_all_classes[event_id] < t:
                events_to_remove[incident_type].append(event_id)

    return events_to_remove


def remove_event_from_incidents(data_df, events_to_remove_per_incident, allowed_event=None):
    def remove_events(row, events_to_remove_per_incident, column_names, allowed_event=None):
        incident_type = row.iloc[-1]
        events_to_remove = set(events_to_remove_per_incident[incident_type])

        if allowed_event is not None:
            events_to_remove.discard(allowed_event)

        event_ids = row.loc["events_sequence"]
        keep_mask = ~np.isin(event_ids, list(events_to_remove))

        list_columns = []
        for col in column_names:
            if isinstance(row.loc[col], np.ndarray):
                list_columns.append(row.loc[col])
        list_columns = np.array(list_columns)

        filtered_columns = list_columns[:, keep_mask]
        for i, col in enumerate(column_names[:-1]):
            if col == "train_kph_sequence":
                row.loc[col] = filtered_columns[i].astype(float)
            else:
                row.loc[col] = filtered_columns[i].astype(int)
        return row

    filtered_data_df = data_df.copy()
    filtered_data_df = filtered_data_df.apply(
        lambda row: remove_events(row, events_to_remove_per_incident, data_df.columns, allowed_event), axis=1)

    return filtered_data_df


def filter_events_out_of_interval(data_df, interval):
    def filter_events(row, interval, columns_names):
        seconds_to_incident = row.loc["seconds_to_incident_sequence"]
        too_low = len(seconds_to_incident[seconds_to_incident < interval[0]])
        too_high = len(seconds_to_incident[seconds_to_incident > interval[1]])
        for col in columns_names:
            if isinstance(row.loc[col], np.ndarray):
                row.loc[col] = row.loc[col][too_low:][:-too_high]
        return row

    columns_names = data_df.columns
    filtered_data_df = data_df.apply(lambda row: filter_events(row, interval, columns_names), axis=1)
    return filtered_data_df


def remove_short_rows(row, x):
    try:
        return len(row) > x
    except:
        return False


def filter_irrelevant_events(data_df, t, allowed_event=None):
    events_sequences_all_classes = np.concatenate(list(data_df["events_sequence"]))
    events_freq_all_classes = compute_events_freq(events_sequences_all_classes)
    events_to_remove_per_incident = compute_events_to_remove(data_df, events_freq_all_classes, t)
    filtered_data_df = remove_event_from_incidents(data_df, events_to_remove_per_incident, allowed_event).reset_index(
        drop=True)

    filtered_data_df = filtered_data_df[filtered_data_df["events_sequence"].apply(lambda row: remove_short_rows(row, x=2))].reset_index(drop=True)

    return filtered_data_df


