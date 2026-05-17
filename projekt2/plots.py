import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import Voronoi, voronoi_plot_2d

def voronoi(X, labels, title, y_true=None):
    plt.figure(figsize=(10, 8))
    margin_x = (X[:, 0].max() - X[:, 0].min()) * 0.1
    margin_y = (X[:, 1].max() - X[:, 1].min()) * 0.1

    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y

    if len(X) > 0:
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.002), np.arange(y_min, y_max, 0.002))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X, labels)
        Z = knn.predict(grid_points).reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')

        try:
            noise = np.random.normal(0, 1e-5, X.shape)
            X_jittered = X + noise

            vor = Voronoi(X_jittered)
            voronoi_plot_2d(vor, ax=plt.gca(), show_points=False, show_vertices=False,
                            line_colors='black', line_width=1.0, line_alpha=0.6)
        except Exception as e:
            print(f"Nie udało się narysować linii Voronoi (błąd: {e}). Tło 1-NN pozostaje.")

    color_source = y_true if y_true is not None else labels

    scatter = plt.scatter(X[:, 0], X[:, 1], c=color_source, cmap='tab10', edgecolor='k', s=25, zorder=3)

    legend1 = plt.legend(*scatter.legend_elements(), title="Cyfry", loc="upper right")
    plt.gca().add_artist(legend1)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)

    plt.show()


def plot_decision_boundaries(xx, yy, Z, X, y, title="Granice decyzyjne", class_names=None):
    plt.figure(figsize=(11, 8))

    cmap = plt.get_cmap('tab10', 10)

    contour = plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, 10.5, 1))

    scatter = plt.scatter(X[:1000, 0], X[:1000, 1], c=y[:1000],
                          cmap=cmap, edgecolors='k', s=30, alpha=0.8,
                          vmin=-0.5, vmax=9.5)

    plt.title(title, fontsize=14)
    plt.xlabel('Cecha 1 (Suma jasności / Ink)')
    plt.ylabel('Cecha 2 (Symetria)')

    if class_names is None:
        class_names = [f"Klasa {i}" for i in range(10)]

    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Klasy", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_feature_metrics(results):
    names = list(results.keys())

    sil_scores = [metrics['Silhouette'] for metrics in results.values()]
    ari_scores = [metrics['ARI'] for metrics in results.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(x - width / 2, sil_scores, width, label='Silhouette Score (wg etykiet)', color='dodgerblue')
    ax.bar(x + width / 2, ari_scores, width, label='ARI (K-Means)', color='darkorange')

    ax.set_ylabel('Wartość metryki')
    ax.set_title('Ocena separowalności klas w zależności od liczby cech')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    num_classes = cm.shape[0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_classes)))

    disp.plot(cmap='Blues', values_format='d', ax=ax)
    plt.title(title)
    plt.show()



def plot_learning_curves(train_acc_history, test_acc_history, title="Krzywe uczenia"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_acc_history) + 1)

    plt.plot(epochs, train_acc_history, label='Accuracy Treningowe', marker='o')
    plt.plot(epochs, test_acc_history, label='Accuracy Testowe', marker='s')

    plt.title(title)
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()