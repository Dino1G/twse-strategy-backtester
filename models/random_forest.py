import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=50, random_state=42, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba
