from copy import deepcopy
from pathlib import Path

from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File

from flower.data import Dataset, save_dataset
from flower.sampling import load_walker
from flower.utils import _new_file, copy_app_future


@python_app(executors=['default'])
def get_continue_flag(nstates, inputs=[], outputs=[]):
    continue_flag = sum([state is not None for state in inputs]) < nstates
    return continue_flag


@join_app
def conditional_sample(
        context,
        continue_flag,
        walkers,
        biases,
        nstates,
        shuffled_sampling,
        model,
        checks,
        inputs=[],
        outputs=[],
        ):
    import numpy as np
    from flower.data import read_dataset
    from flower.ensemble import get_continue_flag
    from flower.utils import _new_file
    states = inputs
    if (len(states) < len(walkers)) or continue_flag:
        if shuffled_sampling:
            index = np.random.randint(0, len(walkers))
        else:
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
        return conditional_sample(
                context,
                get_continue_flag(nstates, inputs=states),
                walkers,
                biases,
                nstates,
                shuffled_sampling,
                model,
                checks,
                inputs=states,
                outputs=[outputs[0]],
                )
    data_future = context.apps(Dataset, 'save_dataset')(states=None, inputs=states, outputs=[outputs[0]])
    return data_future


@join_app
def reset_walkers(walkers, indices):
    for i, walker in enumerate(walkers):
        if i in indices:
            walker.reset()


class Ensemble:
    """Wraps a set of walkers"""

    def __init__(self, context, walkers, biases=[], shuffled_sampling=False):
        assert len(walkers) > 0
        self.context = context
        self.walkers = walkers
        if len(biases) > 0:
            assert len(biases) == len(walkers)
        else:
            biases = [None] * len(walkers)
        self.biases = biases
        self.shuffled_sampling = shuffled_sampling

    def sample(self, nstates, model=None, checks=None):
        data_future = conditional_sample(
                self.context,
                True,
                self.walkers,
                self.biases,
                nstates,
                shuffled_sampling=self.shuffled_sampling,
                model=model,
                checks=checks if checks is not None else [],
                inputs=[],
                outputs=[File(_new_file(self.context.path, 'data_', '.xyz'))],
                ).outputs[0]
        return Dataset(self.context, data_future=data_future)

    def as_dataset(self, checks=None):
        context = self.walkers[0].context
        states = []
        for i, walker in enumerate(self.walkers):
            state = walker.state_future
            if checks is not None:
                for check in checks:
                    state = check(state, walker.tag_future)
            states.append(state)
        return Dataset( # None states are filtered in constructor
                context,
                atoms_list=states,
                )

    def save(self, path, require_done=True):
        path = Path(path)
        assert path.is_dir()
        for i, (walker, bias) in enumerate(zip(self.walkers, self.biases)):
            path_walker = path / str(i)
            path_walker.mkdir(parents=False, exist_ok=False)
            walker.save(path_walker, require_done=require_done)
            if bias is not None:
                bias.save(path_walker, require_done=require_done)

    def reset(self, indices):
        reset_walkers(self.walkers, indices)


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
    def from_walker(cls, walker, nwalkers, dataset=None):
        """Initialize ensemble based on single walker"""
        walkers = []
        for i in range(nwalkers):
            _walker = walker.copy()
            _walker.parameters.seed = i
            if dataset is not None:
                _walker.state_future = copy_app_future(dataset[i])
                _walker.start_future = copy_app_future(dataset[i])
            walkers.append(_walker)
        return cls(walker.context, walkers)
