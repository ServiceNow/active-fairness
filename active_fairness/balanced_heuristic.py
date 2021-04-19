import numpy as np
from baal.active.heuristics import AbstractHeuristic


class BalancedHeuristic(AbstractHeuristic):
    def __init__(self, al_dataset, base_heuristic, sensitive_attribute):
        super().__init__()
        self.al_dataset = al_dataset
        self.base_heuristic = base_heuristic
        self.sensitive_attribute = sensitive_attribute

    def get_ranks(self, predictions):
        sensitives = np.array([y[self.sensitive_attribute] for _, y in self.al_dataset.pool])
        ranks = self.base_heuristic.get_ranks(predictions)
        sensitives_sorted = sensitives[ranks]
        groups = [ranks[sensitives_sorted == gr] for gr in np.unique(sensitives)]
        interlaced = np.array(list(zip(*groups))).reshape([-1])
        return interlaced
