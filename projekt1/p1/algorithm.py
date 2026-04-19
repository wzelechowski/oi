from abc import ABC, abstractmethod


class Algorithm(ABC):

    @abstractmethod
    def fit_predict(self, data):
        pass

    @abstractmethod
    def get_param_value(self):
        pass

    @abstractmethod
    def get_param_name(self):
        pass

    @abstractmethod
    def get_algorithm_name(self):
        pass