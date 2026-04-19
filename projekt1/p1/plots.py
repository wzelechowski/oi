import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt

def silhouette(values, scores, clusters, filename, x_label):
    plt.figure()
    plt.plot(values, scores, marker='o', label='silhouette', color='b')
    plt.title(f"{filename} - Silhouette Score")
    plt.xlabel(x_label)
    plt.ylabel("Silhouette score")
    for i, n_clust in enumerate(clusters):
        plt.text(values[i], min(-0.1, min(scores) - 0.1), str(n_clust), ha='center', va='bottom', fontsize=10, zorder=3)
    plt.ylim(min(-0.1, min(scores) - 0.1), max(scores) + 0.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def metrics(values, ari_scores, hom_scores, comp_scores, clusters, filename, x_label):
    plt.figure()
    plt.plot(values, ari_scores, label='Adjusted Rand Score', color='blue')
    plt.plot(values, hom_scores, label='Homogeneity', color='orange')
    plt.plot(values, comp_scores, label='Completeness', color='green')

    plt.title(f"{filename} - Miary zewnętrzne")
    plt.xlabel(x_label)

    plt.ylim(-0.1, 1.1)
    if x_label == "n_clusters":
        plt.xticks(clusters)

    for i, n_clust in enumerate(clusters):
        plt.text(values[i], -0.1, str(n_clust), ha='center', va='bottom', fontsize=10, zorder=3)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.show()

def voronoi(X, labels, title, y_true=None):
    plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    if len(X) > 0:
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X, labels)
        Z = knn.predict(grid_points).reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')

        vor = Voronoi(X)
        voronoi_plot_2d(vor, ax=plt.gca(), show_points=False, show_vertices=False,
                        line_colors='black', line_width=1.5, line_alpha=0.7)

    color_source = y_true if y_true is not None else labels
    plt.scatter(X[:, 0], X[:, 1], c=color_source, cmap='Set1', edgecolor='k', s=40, zorder=3)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)

    plt.show()
