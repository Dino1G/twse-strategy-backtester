from .base_model import BaseModel
from sklearn.ensemble import HistGradientBoostingClassifier


class HistGBModel(BaseModel):

    def __init__(self, **kwargs):
        self.model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba
