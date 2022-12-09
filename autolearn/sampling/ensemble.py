from copy import deepcopy

from autolearn import Dataset


class Ensemble:
    """Wraps a set of walkers"""

    def __init__(self, context, walkers):
        self.context = context
        self.walkers = walkers

    def propagate(self, safe_return=False, **kwargs):
        return Dataset(
                self.context,
                atoms_list=[w.propagate(safe_return, **kwargs) for w in self.walkers],
                )

    @property
    def nwalkers(self):
        return len(self.walkers)

    @classmethod
    def from_walker(cls, walker, nwalkers):
        """Initialize ensemble based on single walker"""
        walkers = []
        for i in range(nwalkers):
            _walker = walker.copy()
            _walker.parameters.seed = i
            walkers.append(_walker)
        return cls(walker.context, walkers)
