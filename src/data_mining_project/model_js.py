import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def modified_jaccard_similarity(A, B):
    set_A = set(A)
    set_B = set(B)
    intersection_size = len(set_A.intersection(set_B))
    union_size = len(set_A.union(set_B))

    return intersection_size / union_size


def compute_similarity_to_incident_group(X, Y, x, incident_type):
    incident_group = []
    for i in range(Y.shape[0]):
        if Y[i] == incident_type:
            incident_group.append(X[i])

    similarity_to_incident_group = []
    for i in range(len(incident_group)):
        sim = 0
        if len(X.shape) > 1:
            for col in range(X.shape[1]):
                sim += modified_jaccard_similarity(incident_group[i][col], x[col])
        else:
            sim = modified_jaccard_similarity(incident_group[i], x)
        similarity_to_incident_group.append(np.mean(sim))

    return np.array(similarity_to_incident_group).mean()


def init_confusion_matrix(incident_types):
    class_count = {incident_type: 0 for incident_type in incident_types}
    confusion_matrix = {}
    for incident_type in incident_types:
        confusion_matrix[incident_type] = class_count.copy()

    return confusion_matrix


def too_modified_js(X, Y):
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


def compute_f1_score(confusion_matrix):
    incident_types = list(confusion_matrix.keys())
    matrix = []

    for incident_type1 in incident_types:
        row = []
        for incident_type2 in incident_types:
            row.append(confusion_matrix[incident_type1].get(incident_type2, 0))
        matrix.append(row)

    matrix = np.array(matrix)
    precision = np.zeros(len(incident_types))
    recall = np.zeros(len(incident_types))
    f1_scores = np.zeros(len(incident_types))

    for i in range(len(incident_types)):
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp
        tn = np.sum(matrix) - (tp + fp + fn)

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[
            i]) > 0 else 0

    f1_macro = np.mean(f1_scores)

    return f1_macro, f1_scores


def plot_confusion_matrix(confusion_matrix, t):
    incident_types = list(confusion_matrix.keys())
    matrix = []

    for incident_type1 in incident_types:
        row = []
        for incident_type2 in incident_types:
            row.append(confusion_matrix[incident_type1].get(incident_type2, 0))
        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, cmap="Blues", xticklabels=incident_types, yticklabels=incident_types, fmt=".2f")
    plt.title(f"Confusion Matrix Heatmap | Relevance Threshold: {t}")
    plt.xlabel("Predicted Incident Type")
    plt.ylabel("True Incident Type")
    plt.show()


def plot_f1_scores(f1_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(f1_scores[:, 0], f1_scores[:, 1], marker='o', linestyle='-', color='b', zorder=2)
    max_f1 = max(f1_scores, key=lambda x: x[1])
    plt.scatter(max_f1[0], max_f1[1], color='red', label=f'(t, F1): ({max_f1[0], round(max_f1[1], 3)})', zorder=3)

    plt.xlabel("Relevance Threshold t")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.legend()
    plt.show()
