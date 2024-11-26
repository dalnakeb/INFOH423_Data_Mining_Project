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
    def remove_events(row, list_columns_indices, events_to_remove_per_incident, allowed_event=None):
        incident_type = row.iloc[-1]
        events_to_remove = set(events_to_remove_per_incident[incident_type])

        if allowed_event is not None:
            events_to_remove.discard(allowed_event)  # Faster than `remove`

        event_ids = row.iloc[2]  # Assuming this is the array of event IDs
        keep_mask = ~np.isin(event_ids, list(events_to_remove))  # Vectorized operation

        list_columns = np.array([row.iloc[col] for col in list_columns_indices])
        filtered_columns = list_columns[:, keep_mask]
        for i, col in enumerate(list_columns_indices):
            if i == 3:
                row.iloc[col] = filtered_columns[i].astype(float)
            else:
                row.iloc[col] = filtered_columns[i].astype(int)
        return row

    filtered_data_df = data_df.copy()
    list_columns = ["vehicles_sequence", "events_sequence", "seconds_to_incident_sequence", "train_kph_sequence",
                    "dj_ac_state_sequence", "dj_dc_state_sequence"]
    list_columns_indices = []
    for el in list_columns:
        list_columns_indices.append(data_df.columns.to_list().index(el))

    filtered_data_df = filtered_data_df.apply(
        lambda row: remove_events(row, list_columns_indices, events_to_remove_per_incident, allowed_event), axis=1)
    return filtered_data_df


def filter_events(data_df, t, allowed_event=None):
    events_sequences_all_classes = np.concatenate(list(data_df["events_sequence"]))
    events_freq_all_classes = compute_events_freq(events_sequences_all_classes)
    events_to_remove_per_incident = compute_events_to_remove(data_df, events_freq_all_classes, t)
    filtered_data_df = remove_event_from_incidents(data_df, events_to_remove_per_incident, allowed_event).reset_index(
        drop=True)
    return filtered_data_df
