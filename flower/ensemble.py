from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Any, Dict
import typeguard
from copy import deepcopy
from pathlib import Path

from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from flower.execution import ExecutionContext
from flower.models import BaseModel
from flower.checks import Check
from flower.data import Dataset, save_dataset, FlowerAtoms
from flower.sampling import load_walker, BaseWalker, PlumedBias
from flower.utils import _new_file, copy_app_future


@typeguard.typechecked
def _count_nstates(
        inputs: List[Optional[FlowerAtoms]] = [],
        outputs: List[File] = [],
        ) -> int:
    return sum([state is not None for state in inputs])
count_nstates = python_app(_count_nstates, executors=['default'])


@join_app
@typeguard.typechecked
def conditional_sample(
        context: ExecutionContext,
        nstates: int,
        nstates_effective: Union[int, AppFuture],
        walkers: List[BaseWalker],
        biases: List[Optional[PlumedBias]],
        model: Optional[BaseModel], # None for e.g. RandomWalker
        checks: List[Check],
        inputs: List[Optional[FlowerAtoms]] = [],
        outputs: List[File] = [],
        ): # recursive
    import numpy as np
    from flower.data import read_dataset
    from flower.ensemble import count_nstates
    from flower.utils import _new_file
    states = inputs
    if nstates_effective == 0:
        for i in range(len(walkers)):
            index = i # no shuffle
            walker = walkers[index]
            bias   = biases[index]
            state = walker.propagate(
                    safe_return=False,
                    bias=bias,
                    keep_trajectory=False,
                    )
            walker.parameters.seed += len(walkers) # avoid generating same states
            for check in checks:
                state = check(state, walker.tag_future)
            walker.reset_if_unsafe()
            states.append(state) # some are None
    else:
        batch_size = nstates - nstates_effective
        if not batch_size > 0:
            assert batch_size == 0 # cannot be negative
            data_future = context.apps(Dataset, 'save_dataset')(
                    states=None,
                    inputs=states,
                    outputs=[outputs[0]],
                    )
            return data_future
        for i in range(batch_size):
            index = np.random.randint(0, len(walkers))
            walker = walkers[index]
            bias  = biases[index]
            state = walker.propagate(
                    safe_return=False,
                    bias=bias,
                    keep_trajectory=False,
                    )
            walker.parameters.seed += len(walkers) # avoid generating same states
            for check in checks:
                state = check(state, walker.tag_future)
            walker.reset_if_unsafe()
            states.append(state) # some are None
    return conditional_sample(
            context,
            nstates,
            count_nstates(inputs=states),
            walkers,
            biases,
            model,
            checks,
            inputs=states,
            outputs=[outputs[0]],
            )


@join_app
@typeguard.typechecked
def reset_walkers(
        walkers: List[BaseWalker],
        indices: Union[List[int], AppFuture],
        ) -> AppFuture:
    for i, walker in enumerate(walkers):
        if i in indices:
            future = walker.reset()
    return future # return last future to enforce execution


@typeguard.typechecked
class Ensemble:
    """Wraps a set of walkers"""

    def __init__(
            self,
            context: ExecutionContext,
            walkers: List[BaseWalker],
            biases: Optional[List[Optional[PlumedBias]]] = None,
            ) -> None:
        assert len(walkers) > 0
        self.context = context
        self.walkers = walkers
        if biases is not None:
            assert len(biases) == len(walkers)
        else:
            biases = [None] * len(walkers)
        self.biases = biases

    def sample(
            self,
            nstates: int,
            model: Optional[BaseModel] = None,
            checks: Optional[List[Check]] = None,
            ) -> Dataset:
        data_future = conditional_sample(
                self.context,
                nstates,
                0,
                self.walkers,
                self.biases,
                model=model,
                checks=checks if checks is not None else [],
                inputs=[],
                outputs=[File(_new_file(self.context.path, 'data_', '.xyz'))],
                ).outputs[0]
        return Dataset(self.context, None, data_future=data_future)

    def as_dataset(self, checks: Optional[List[Check]] = None) -> Dataset:
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

    def save(self, path: Union[Path, str], require_done: bool = True) -> None:
        path = Path(path)
        assert path.is_dir()
        for i, (walker, bias) in enumerate(zip(self.walkers, self.biases)):
            path_walker = path / str(i)
            path_walker.mkdir(parents=False, exist_ok=False)
            walker.save(path_walker, require_done=require_done)
            if bias is not None:
                bias.save(path_walker, require_done=require_done)

    def reset(self, indices: Union[List[int], AppFuture]) -> AppFuture:
        return reset_walkers(self.walkers, indices)

    @classmethod
    def load(cls, context: ExecutionContext, path: Union[Path, str]) -> Ensemble:
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
    def nwalkers(self) -> int:
        return len(self.walkers)

    @classmethod
    def from_walker(
            cls,
            walker: BaseWalker,
            nwalkers: int,
            dataset: Optional[Dataset] = None,
            ):
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
