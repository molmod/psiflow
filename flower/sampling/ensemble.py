from copy import deepcopy

from flower.data import Dataset


class Ensemble:
    """Wraps a set of walkers"""

    def __init__(self, context, walkers, biases=[], global_bias=None):
        self.context = context
        self.walkers = walkers
        if len(biases) > 0:
            assert len(biases) == len(walkers)
        else:
            biases = [None] * len(walkers)
        self.biases = biases
        self.global_bias = global_bias

    def propagate(self, safe_return=False, **kwargs):
        iterator = zip(self.walkers, self.biases)
        atoms_list = [w.propagate(safe_return, bias=b, **kwargs) for w, b in iterator]
        return Dataset(
                self.context,
                atoms_list=atoms_list,
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
