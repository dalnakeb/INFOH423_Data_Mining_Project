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