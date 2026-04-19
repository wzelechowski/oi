from sklearn.cluster import DBSCAN
from algorithm import Algorithm

class DBScan(Algorithm):
    def __init__(self, eps=0.5, min_samples=1, **kwargs):
        self.model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)

    def fit_predict(self, data):
        return self.model.fit_predict(data)

    def get_param_value(self):
        return self.model.eps

    def get_param_name(self):
        return "eps"

    def get_algorithm_name(self):
        return "DBSCAN"