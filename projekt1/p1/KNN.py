from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self):
        pass

    def get_algorithm_name(self):
        return f"K-NN"

    def get_param_name(self):
        return "n_neighbors"

    def get_param_value(self):
        return self.n_neighbors