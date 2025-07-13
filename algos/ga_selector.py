import random
import numpy as np
from deap import base, creator, tools, algorithms
from .base_selector import BaseSelector
from models.random_forest import RandomForestModel


class GASelector(BaseSelector):
    def __init__(self, pop_size=10, ngen=1, days=10, seed=42):
        self.pop_size = pop_size
        self.ngen = ngen
        self.days = days
        random.seed(seed)
        np.random.seed(seed)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()

    def _rolling_accuracy(self, data, feats, label_col):
        dates = data.index.sort_values()
        last_days = dates[-(self.days+1):]
        accs = []
        for d in last_days[1:]:
            train = data.loc[data.index < d]
            test = data.loc[[d]]
            clf = RandomForestModel(n_estimators=50, random_state=42)
            clf.fit(train[feats], train[label_col])
            pred = clf.predict(test[feats])[0]
            accs.append(int(pred == test[label_col].iloc[0]))
        return np.mean(accs)

    def fit(self, data, features, label):
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat,
                              creator.Individual, self.toolbox.attr_bool,
                              n=len(features))
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        eval_hist = []

        def eval_ind(ind):
            sel = [f for bit, f in zip(ind, features) if bit]
            score = self._rolling_accuracy(data, sel, label)
            eval_hist.append((ind.copy(), score))
            return (score,)

        self.toolbox.register("evaluate", eval_ind)
        self.toolbox.register("mate",    tools.cxTwoPoint)
        self.toolbox.register("mutate",  tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select",  tools.selTournament, tournsize=3)

        pop = self.toolbox.population(n=self.pop_size)
        algorithms.eaSimple(pop, self.toolbox,
                            cxpb=0.5, mutpb=0.2,
                            ngen=self.ngen, verbose=True)

        selected = []
        for ind, fit in eval_hist:
            if fit >= 0.1:
                feats = [f for bit, f in zip(ind, features) if bit]
                selected.append(feats)
        return selected
