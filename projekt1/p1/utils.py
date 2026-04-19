import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.preprocessing import StandardScaler

import plots

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=';')
    numeric_data = df.select_dtypes(include=["number"])

    X = numeric_data.iloc[:, :-1].to_numpy()
    y_true = numeric_data.iloc[:, -1].to_numpy()

    X_scaled = StandardScaler().fit_transform(X)

    return X_scaled, y_true


def evaluate_algorithms(X_scaled, algorithms):
    scores = []
    all_labels = []
    best_model = None
    best_score = -1
    best_labels = None

    for algo in algorithms:
        labels = algo.fit_predict(X_scaled)
        all_labels.append(labels)

        unique_labels = np.unique(labels)
        if 1 < len(unique_labels) < len(X_scaled):
            score = silhouette_score(X_scaled, labels)
        else:
            score = -1

        scores.append(score)

        if score > best_score:
            best_score = score
            best_model = algo
            best_labels = labels

    return scores, best_model, best_labels, all_labels

def evaluate_metrics(all_labels, y_true):
    ari_scores = []
    hom_scores = []
    comp_scores = []
    for labels in all_labels:
        ari_scores.append(adjusted_rand_score(y_true, labels))
        hom_scores.append(homogeneity_score(y_true, labels))
        comp_scores.append(completeness_score(y_true, labels))

    return ari_scores, hom_scores, comp_scores

def draw_plots(X, y_true, algorithm, file):
    scores, best_model, best_labels, all_labels = evaluate_algorithms(X, algorithm)
    ari_scores, hom_scores, comp_scores = evaluate_metrics(all_labels, y_true)
    values = [algo.get_param_value() for algo in algorithm]

    clusters = [len(set(labels)) for labels in all_labels]

    plots.silhouette(values, scores, clusters, file, algorithm[0].get_param_name())

    plots.metrics(values, ari_scores, hom_scores, comp_scores, clusters, file, algorithm[0].get_param_name())
    valid_indices = [i for i, score in enumerate(scores)]
    if valid_indices:
        param = algorithm[0].get_param_name()
        name = algorithm[0].get_algorithm_name()

        best_idx_1 = max(valid_indices, key=lambda i: scores[i])
        worst_idx_1 = min(valid_indices, key=lambda i: scores[i])
        best_idx_2 = max(valid_indices, key=lambda i: ari_scores[i])
        worst_idx_2 = min(valid_indices, key=lambda i: ari_scores[i])

        if X.shape[1] == 2:
            plots.voronoi(X, all_labels[best_idx_1], f"{file} - Exp 1: NAJLEPSZY {name} ({param}={values[best_idx_1]})")
            plots.voronoi(X, all_labels[worst_idx_1],
                          f"{file} - Exp 1: NAJGORSZY {name} ({param}={values[worst_idx_1]})")
            plots.voronoi(X, all_labels[best_idx_2], f"{file} - Exp 2: NAJLEPSZY {name} ({param}={values[best_idx_2]})",
                          y_true=y_true)
            plots.voronoi(X, all_labels[worst_idx_2],
                          f"{file} - Exp 2: NAJGORSZY {name} ({param}={values[worst_idx_2]})", y_true=y_true)

    else:
        print(f"[{file}] Brak modeli spełniających warunki (minimum 2 klastry) do narysowania diagramu.")
