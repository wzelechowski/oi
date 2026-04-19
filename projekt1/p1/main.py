import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler

from dbscan import DBScan
from kmeans import Kmeans
from utils import load_and_preprocess_data, draw_plots


def main():
    load_random_sets()
    load_reals_sets()

def load_random_sets():
    files = ["data_1/2_1.csv", "data_1/2_2.csv", "data_1/2_3.csv", "data_1/3_1.csv", "data_1/3_2.csv", "data_1/3_3.csv"]

    eps_range = np.round(np.arange(0.1, 1, 0.1), 2)

    for file in files:
        X_scaled, y_true = load_and_preprocess_data(file)

        kmeans_strategies = [Kmeans(n_clusters=k) for k in range(2, 10)]
        draw_plots(X_scaled, y_true, kmeans_strategies, file)

        dbscan_strategies = [DBScan(eps=eps, min_samples=1) for eps in eps_range]
        draw_plots(X_scaled, y_true, dbscan_strategies, file)

def load_reals_sets():
    eps_range = np.round(np.arange(0.1, 3, 0.3), 2)

    iris = load_iris()
    X_iris = StandardScaler().fit_transform(iris.data)
    y_iris = iris.target

    kmeans_iris = [Kmeans(n_clusters=k) for k in range(2, 10)]
    draw_plots(X_iris, y_iris, kmeans_iris, "Iris")

    dbscan_iris = [DBScan(eps=eps, min_samples=1) for eps in eps_range]
    draw_plots(X_iris, y_iris, dbscan_iris, "Iris")

    wine = load_wine()
    X_wine = StandardScaler().fit_transform(wine.data)
    y_wine = wine.target

    kmeans_wine = [Kmeans(n_clusters=k) for k in range(2, 10)]
    draw_plots(X_wine, y_wine, kmeans_wine, "Wine")

    dbscan_wine = [DBScan(eps=eps, min_samples=1) for eps in eps_range]
    draw_plots(X_wine, y_wine, dbscan_wine, "Wine")

if __name__ == '__main__':
    main()