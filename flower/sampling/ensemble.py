from copy import deepcopy
from pathlib import Path

from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File

from flower.data import Dataset, save_dataset
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
    from flower.sampling.ensemble import get_continue_flag
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

    def propagate(self, nstates, model=None, checks=[]):
        assert nstates >= len(self.walkers)
        data_future = conditional_propagate(
                self.context,
                True,
                self.walkers,
                self.biases,
                nstates,
                model=model,
                checks=checks,
                inputs=[],
                outputs=[File(_new_file(self.context.path, 'data_', '.xyz'))],
                ).outputs[0]
        return Dataset(self.context, data_future=data_future)

    def save(self, path_dir, require_done=True):
        path_dir = Path(path_dir)
        assert path_dir.is_dir()
        for i, (walker, bias) in enumerate(zip(self.walkers, self.biases)):
            path_start = path_dir / '{}_start.xyz'.format(i)
            path_state = path_dir / '{}_state.xyz'.format(i)
            path_pars  = path_dir / '{}_pars.yaml'.format(i)
            walker.save(path_start, path_state, path_pars, require_done=require_done)
            if bias is not None:
                path_plumed = path_dir / '{}_plumed_input.txt'.format(i)
                paths_data = {}
                for key in bias.data_futures.keys():
                    paths_data[key] = path_dir / '{}_plumed_{}.txt'.format(i, key)
                bias.save(path_plumed, require_done=require_done, **paths_data)

    @classmethod
    def load(cls, context, walker_cls, path_dir):
        path_dir = Path(path_dir)
        assert path_dir.is_dir()
        walkers = []
        biases = []
        i = 0
        while (path_dir / '{}_start.xyz'.format(i)).is_file():
            path_start = path_dir / '{}_start.xyz'.format(i)
            path_state = path_dir / '{}_state.xyz'.format(i)
            path_pars  = path_dir / '{}_pars.yaml'.format(i)
            path_plumed = path_dir / '{}_plumed_input.txt'.format(i)
            walker = walker_cls.load(
                    walker_cls,
                    path_start,
                    path_state,
                    path_pars,
                    )
            if path_plumed.is_file():
                paths_data = {}
                for key in PlumedBias.keys_with_future:
                    path = path_dir / '{}_plumed_{}.txt'.format(i, key)
                    if path.is_file():
                        paths_data[key] = path
                bias = PlumedBias.load(context, path_plumed, **paths_data)
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
