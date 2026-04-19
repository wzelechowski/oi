from sklearn.cluster import KMeans
from algorithm import Algorithm

class Kmeans(Algorithm):
    def __init__(self, n_clusters=3, **kwargs):
        self.model = KMeans(n_clusters=n_clusters, **kwargs)

    def fit_predict(self, data):
        return self.model.fit_predict(data)

    def get_param_value(self):
        return self.model.n_clusters

    def get_param_name(self):
        return "k"

    def get_algorithm_name(self):
        return "KMeans"