import numpy as np
import  matplotlib.pyplot as plt

# Implementing the Smith Waterman local alignment, changed a little using the paper:
# Okada, D., Ino, F. & Hagihara, K. Accelerating the Smith-Waterman algorithm with interpair pruning and band optimization for the all-pairs comparison of base sequences. BMC Bioinformatics 16, 321 (2015). https://doi.org/10.1186/s12859-015-0744-4

def smith_waterman_optimized(seq1: list[str], seq2: list[str]) -> (list[str], list[str]):
    """
    implements the smith waterman alignment algorithm. Assuming the gap penalty is 0 and the mismatch penality is -1
    :param seq1: first sequence
    :param seq2: second sequence
    :return: (sequence 1 aligned, sequence 2 aligned)
    """
    gap_penalty = 0
    match = 1
    mismatch = -1

    row, col = len(seq1) + 1, len(seq2) + 1
    matrix_filling = np.zeros((row, col), dtype=np.int64)
    matrix_tracing = np.zeros((row, col), dtype=np.int64)

    max_score = -1
    max_index = (-1, -1)

    for i in range(1, row):
        for j in range(1, col):
            S = match if seq1[i - 1] == seq2[j - 1] else mismatch
            H = matrix_filling[i - 1, j - 1] + S
            F = matrix_filling[i - 1, j] + gap_penalty
            E = matrix_filling[i, j - 1] + gap_penalty
            max_value = max(0, H, F, E)

            matrix_filling[i, j] = max_value

            if max_value == 0:
                matrix_tracing[i, j] = 0
            elif max_value == H:
                matrix_tracing[i, j] = 3
            elif max_value == F:
                matrix_tracing[i, j] = 2
            else:
                matrix_tracing[i, j] = 1

            if max_value > max_score:
                max_score = max_value
                max_index = (i, j)

    aligned_seq1 = []
    aligned_seq2 = []
    max_i, max_j = max_index

    while matrix_tracing[max_i, max_j] != 0:
        if matrix_tracing[max_i, max_j] == 3:
            aligned_seq1.append(seq1[max_i - 1])
            aligned_seq2.append(seq2[max_j - 1])
            max_i -= 1
            max_j -= 1
        elif matrix_tracing[max_i, max_j] == 2:
            aligned_seq1.append(seq1[max_i - 1])
            aligned_seq2.append(-1)
            max_i -= 1
        elif matrix_tracing[max_i, max_j] == 1:
            aligned_seq1.append(-1)
            aligned_seq2.append(seq2[max_j - 1])
            max_j -= 1

    aligned_seq1.reverse()
    aligned_seq2.reverse()

    return aligned_seq1, aligned_seq2


def sequence_in_common(seq1: list[str], seq2: list[str]) -> (list[str], int):
    """
    For a given two aligned sequences, find the common sub-sequence
    :param seq1: first sequence
    :param seq2: second sequence
    :return: (common sub sequences, length of common sub-sequence)
    """
    if len(seq1) != len(seq2):
        print("The sequences are not the same size!!")
    else:
        seq_in_com = []
        for i in range(len(seq1)):
            if seq1[i] == seq2[i]:
                seq_in_com.append(seq1[i])
    return seq_in_com, len(seq_in_com)


def remove_consecutive_dup(seq: list[str]) -> list[str]:
    """
    For a given sequence, remove any consecutive repetition of the same value.
    ex: (1,1,2,3,1,3,4) -> (1,2,3,1,3,4)
    :param seq:
    :return: sequence with removed consecutive repetitions
    """
    result = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            result.append(seq[i])

    return np.array(result)



def genetic_similarity(seq1: list[str], seq2: list[str]) -> (list[str], float):
    """
    Given two sequences, compute their common sub-sequence with the normalized alignment score using the
    smith-waterman algorithm.
    :param seq1: first sequence
    :param seq2: second sequence
    :return: (common sub-sequence, normalized alignment score)
    """
    seq1 = list(set(seq1))
    seq2 = list(set(seq2))
    seq_a_1, seq_a_2 = smith_waterman_optimized(seq1, seq2)
    seq_com, score = sequence_in_common(seq_a_1, seq_a_2)
    max_len = max(len(seq1), len(seq2))
    return score / max_len


def compute_genetic_similarity_to_incident_group(X: np.array, Y: np.array, x: np.array, incident_type: int) -> float:
    """
    Compute the average genetic similarty of a sequence of events, to a group of event sequences associated to an
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
                sim += genetic_similarity(incident_group[i][col], x[col])
        else:
            sim = genetic_similarity(incident_group[i], x)
        similarity_to_incident_group.append(np.mean(sim))

    return np.array(similarity_to_incident_group).mean()


def init_confusion_matrix(incident_types: np.array) -> dict:
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



def loo_sw(X: np.array, Y: np.array) -> dict:
    """
    Perform a leave-one-out cross-validation of the genetic similarity algorithm to compute the confusion matrix of
    predictions between the true class and the prediction class
    :param X: Sequences of events
    :param Y: Incident types
    :return: Confusion matrix dictionary (incident_type: dictionary(incident_type: num predictions))
    """
    incident_types = np.unique(Y)
    confusion_matrix = init_confusion_matrix(incident_types)
    for i, x in enumerate(X):
        if i % 1 == 0:
            print(i)

        X_minus_i = np.delete(X, i, axis=0)
        Y_minus_i = np.delete(Y, i, axis=0)
        y = Y[i]
        similarity_to_incident_group = []
        for incident_type in incident_types:
            similarity_to_incident_group.append(
                (incident_type, compute_genetic_similarity_to_incident_group(X_minus_i, Y_minus_i, x, incident_type)))
        y_hat = max(similarity_to_incident_group, key=lambda x: x[1])[0]
        confusion_matrix[y][y_hat] += 1

    for c in confusion_matrix:
        incident_group_size = sum([v for v in confusion_matrix[c].values()])
        new_c = {k: v / incident_group_size for k, v in confusion_matrix[c].items()}
        confusion_matrix[c] = new_c.copy()

    return confusion_matrix