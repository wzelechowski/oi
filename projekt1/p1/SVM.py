from sklearn.svm import SVC

from algorithm import Algorithm


class SVM(Algorithm):
    def __init__(self, kernel="rbf", C=1.0):
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=self.kernel, C=self.C, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, data):
        pass

    def get_algorithm_name(self):
        return f"SVM ({self.kernel})"

    def get_param_name(self):
        return "C"

    def get_param_value(self):
        return self.C