from copy import deepcopy
import covalent as ct

from autolearn import Dataset


class Ensemble:
    """Wraps a set of walkers"""

    def __init__(self, walkers):
        self.walkers = walkers

    def propagate(self, model,  model_execution):
        for i in range(self.nwalkers):
            self.walkers[i] = self.walkers[i].propagate(model, model_execution)

    @ct.electron
    @ct.lattice
    def sample(self):
        return Dataset([w.sample() for w in self.walkers])

    @property
    def nwalkers(self):
        return len(self.walkers)

    @classmethod
    def from_walker(cls, walker, nwalkers):
        """Initialize ensemble based on single walker"""
        walkers = []
        for i in range(nwalkers):
            _walker = deepcopy(walker)
            _walker.parameters.seed = i
            walkers.append(_walker)
        return cls(walkers)
