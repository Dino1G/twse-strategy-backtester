from abc import ABC, abstractmethod


class BaseSelector(ABC):

    @abstractmethod
    def fit(self, data, features, label):
        pass
