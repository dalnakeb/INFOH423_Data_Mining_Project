import numpy as np


def jaccard_similarity(A: np.array, B: np.array) -> float:
    """
    Compute the jaccard similarity between two sequences of data, transforming them into two sets.
    :param A: first sequence
    :param B: second sequence
    :return: jaccard similarity
    """
    set_A = set(A)
    set_B = set(B)
    intersection_size = len(set_A.intersection(set_B))
    union_size = len(set_A.union(set_B))

    return intersection_size / union_size


def compute_similarity_to_incident_group(X, Y, x, incident_type):
    """
    Compute the average jaccard similarty of a sequence of events, to a group of event sequences associated to an
    incident type.
    :param X: Events sequences
    :param Y: incident types
    :param x: event sequence with unknown incident type
    :param incident_type:
    :return: the mean similarity to a given incident type group of events sequences
    """
    incident_group = []
    for i in range(Y.shape[0]):
        if Y[i] == incident_type:
            incident_group.append(X[i])

    similarity_to_incident_group = []
    for i in range(len(incident_group)):
        sim = 0
        if len(X.shape) > 1:
            for col in range(X.shape[1]):
                sim += jaccard_similarity(incident_group[i][col], x[col])
        else:
            sim = jaccard_similarity(incident_group[i], x)
        similarity_to_incident_group.append(np.mean(sim))

    return np.array(similarity_to_incident_group).mean()


def init_confusion_matrix(incident_types):
    """
    Initialize an empty confusion matrix given the different classes
    :param incident_types:
    :return: Confusion matrix dictionary (incident_type: dictionary(incident_type: num predictions))
    """
    class_count = {incident_type: 0 for incident_type in incident_types}
    confusion_matrix = {}
    for incident_type in incident_types:
        confusion_matrix[incident_type] = class_count.copy()

    return confusion_matrix


def loo_js(X, Y):
    """
    Perform a leave-one-out cross-validation of the jaccard similarity algorithm to compute the confusion matrix of
    predictions between the true class and the prediction class
    :param X: Sequences of events
    :param Y: Incident types
    :return: Confusion matrix dictionary (incident_type: dictionary(incident_type: num predictions))
    """
    incident_types = np.unique(Y)
    confusion_matrix = init_confusion_matrix(incident_types)
    for i, x in enumerate(X):
        if i % 50 == 0:
            print(i)

        X_minus_i = np.delete(X, i, axis=0)
        Y_minus_i = np.delete(Y, i, axis=0)
        y = Y[i]
        similarity_to_incident_group = []
        for incident_type in incident_types:
            similarity_to_incident_group.append(
                (incident_type, compute_similarity_to_incident_group(X_minus_i, Y_minus_i, x, incident_type)))
        y_hat = max(similarity_to_incident_group, key=lambda x: x[1])[0]
        confusion_matrix[y][y_hat] += 1

    for c in confusion_matrix:
        incident_group_size = sum([v for v in confusion_matrix[c].values()])
        new_c = {k: v / incident_group_size for k, v in confusion_matrix[c].items()}
        confusion_matrix[c] = new_c.copy()

    return confusion_matrix


