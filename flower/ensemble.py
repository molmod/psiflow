from copy import deepcopy
from pathlib import Path

from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File

from flower.data import Dataset, save_dataset
from flower.sampling import load_walker
from flower.utils import _new_file


@python_app
def get_continue_flag(nstates, inputs=[], outputs=[]):
    from flower.data import read_dataset
    continue_flag = sum([state is not None for state in inputs]) < nstates
    return continue_flag


@join_app
def conditional_propagate(
        context,
        continue_flag,
        walkers,
        biases,
        nstates,
        model,
        checks,
        inputs=[],
        outputs=[],
        ):
    from flower.data import read_dataset
    from flower.ensemble import get_continue_flag
    from flower.utils import _new_file
    states = inputs
    if (len(states) < len(walkers)) or continue_flag:
        index = int(len(states) % len(walkers))
        walker = walkers[index]
        bias   = biases[index]
        state = walker.propagate(
                safe_return=False,
                bias=bias,
                keep_trajectory=False,
                )
        walker.reset_if_unsafe()
        walker.parameters.seed += len(walkers) # avoid generating same states
        for check in checks:
            state = check(state, walker.tag_future)
        states.append(state) # some are None
        return conditional_propagate(
                context,
                get_continue_flag(nstates, inputs=states),
                walkers,
                biases,
                nstates,
                model,
                checks,
                inputs=states,
                outputs=[outputs[0]],
                )
    data_future = context.apps(Dataset, 'save_dataset')(states=None, inputs=states, outputs=[outputs[0]])
    return data_future


class Ensemble:
    """Wraps a set of walkers"""

    def __init__(self, context, walkers, biases=[]):
        assert len(walkers) > 0
        self.context = context
        self.walkers = walkers
        if len(biases) > 0:
            assert len(biases) == len(walkers)
        else:
            biases = [None] * len(walkers)
        self.biases = biases

    def propagate(self, nstates, model=None, checks=None):
        assert nstates >= len(self.walkers)
        data_future = conditional_propagate(
                self.context,
                True,
                self.walkers,
                self.biases,
                nstates,
                model=model,
                checks=checks if checks is not None else [],
                inputs=[],
                outputs=[File(_new_file(self.context.path, 'data_', '.xyz'))],
                ).outputs[0]
        return Dataset(self.context, data_future=data_future)

    def save(self, path, require_done=True):
        path = Path(path)
        assert path.is_dir()
        for i, (walker, bias) in enumerate(zip(self.walkers, self.biases)):
            path_walker = path / str(i)
            path_walker.mkdir(parents=False, exist_ok=False)
            walker.save(path_walker, require_done=require_done)
            if bias is not None:
                bias.save(path_walker, require_done=require_done)

    @classmethod
    def load(cls, context, path):
        path = Path(path)
        assert path.is_dir()
        walkers = []
        biases = []
        i = 0
        while (path / str(i)).is_dir():
            path_walker = path / str(i)
            walker = load_walker(context, path_walker)
            path_plumed = path_walker / 'plumed_input.txt'
            if path_plumed.is_file(): # check if bias present
                bias = PlumedBias.load(context, path_walker)
            else:
                bias = None
            walkers.append(walker)
            biases.append(bias)
            i += 1
        return cls(context, walkers, biases)

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


def generate_distributed_ensemble(walker, bias, cv_name, cv_grid, dataset=None):
    if dataset is None: # explore CV space manually!
        raise NotImplementedError
    else:
        pass
