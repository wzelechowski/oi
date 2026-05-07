from sklearn.neural_network import MLPClassifier

from p1.algorithm import Algorithm


class MLP(Algorithm):
    def __init__(self, activation="relu", hidden_layer_sizes=(10,0), random_state=42):
        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            max_iter=100000,
            tol=0,
            n_iter_no_change=100000,
            solver='sgd',
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def fit_predict(self, data):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def partial_fit(self, X, y, classes):
        return self.model.partial_fit(X, y, classes=classes)

    def get_algorithm_name(self):
        return f"MLP ({self.activation})"

    def get_param_name(self):
        return "hidden_layer_sizes"

    def get_param_value(self):
        return self.hidden_layer_sizes[0]
