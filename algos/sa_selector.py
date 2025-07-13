import random
import math
import numpy as np
from deap import base, creator, tools, algorithms
from .base_selector import BaseSelector
from models.random_forest import RandomForestModel


class SASelector(BaseSelector):

    def __init__(self, days=10, iter_max=1, init_temp=1.0, alpha=0.9, seed=42):
        self.days = days
        self.iter_max = iter_max
        self.temp = init_temp
        self.alpha = alpha
        random.seed(seed)
        np.random.seed(seed)

    def _rolling_accuracy(self, data, feats, label):
        dates = data.index.sort_values()
        last_days = dates[-(self.days+1):]
        accs = []
        for d in last_days[1:]:
            train = data.loc[data.index < d]
            test = data.loc[[d]]
            clf = RandomForestModel(n_estimators=50, random_state=42)
            clf.fit(train[feats], train[label])
            pred = clf.predict(test[feats])[0]
            accs.append(int(pred == test[label].iloc[0]))
        return np.mean(accs)

    def fit(self, data, features, label):
        best_mask = [random.randint(0, 1) for _ in features]
        best_score = self._rolling_accuracy(
            data, [f for bit, f in zip(best_mask, features) if bit], label)
        cur_mask = best_mask.copy()
        cur_score = best_score

        for i in range(self.iter_max):
            cand = cur_mask.copy()
            idx = random.randrange(len(features))
            cand[idx] = 1 - cand[idx]
            cand_feats = [f for bit, f in zip(cand, features) if bit]
            score = self._rolling_accuracy(data, cand_feats, label)

            if score > cur_score or random.random() < math.exp((score - cur_score)/self.temp):
                cur_mask = cand
                cur_score = score

            if cur_score > best_score:
                best_mask = cur_mask.copy()
                best_score = cur_score

            self.temp *= self.alpha

        selected = [f for bit, f in zip(best_mask, features) if bit]
        return [selected]
