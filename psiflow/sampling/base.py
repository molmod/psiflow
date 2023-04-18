from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, Any
import typeguard
from dataclasses import dataclass, asdict
from copy import deepcopy
import numpy as np
from pathlib import Path

from ase import Atoms

from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.models import BaseModel
from psiflow.utils import copy_app_future, unpack_i, copy_data_future, \
        save_yaml
from psiflow.data import save_atoms, FlowAtoms, Dataset, app_reset_atoms


@typeguard.typechecked
def _conditional_reset(
        state: FlowAtoms,
        start: FlowAtoms,
        tag: str,
        counter: int,
        conditional: bool,
        ) -> tuple[FlowAtoms, str, int]:
    from copy import deepcopy # copy necessary!
    if (not conditional): # reset anyway
        return deepcopy(start), 'safe', 0
    else: # reset if unsafe
        if tag == 'unsafe':
            return deepcopy(start), 'safe', 0
    return deepcopy(state), tag, counter
conditional_reset = python_app(_conditional_reset, executors=['default'])


@typeguard.typechecked
def _is_reset(counter: int) -> bool:
    return counter == 0
is_reset = python_app(_is_reset, executors=['default'])


@typeguard.typechecked
def _sum_counters(counter0: int, counter1: int) -> int:
    return counter0 + counter1
sum_counters = python_app(_sum_counters, executors=['default'])


@typeguard.typechecked
def _update_tag(tag0: str, tag1: str) -> str:
    if tag0 == 'unsafe':
        return 'unsafe'
    else:
        if tag1 == 'unsafe':
            return 'unsafe'
    return 'safe'
update_tag = python_app(_update_tag, executors=['default'])


@typeguard.typechecked
class BaseWalker:

    def __init__(self,
            atoms: Union[Atoms, FlowAtoms, AppFuture],
            seed: int = 0,
            ) -> None:
        if type(atoms) == Atoms:
            atoms = FlowAtoms.from_atoms(atoms)
        self.start_future   = app_reset_atoms(atoms)
        self.state_future   = app_reset_atoms(atoms)
        self.tag_future     = copy_app_future('safe')
        self.counter_future = copy_app_future(0) # counts nsteps
        self.seed = seed
        # apps for walkers are only created at the time when they are invoked,
        # as the execution definition of the model needs to be available.

    def _propagate(self, model):
        raise NotImplementedError

    def propagate(
            self,
            keep_trajectory: bool = False,
            reset_if_unsafe: bool = True,
            model: Optional[BaseModel] = None,
            ) -> Union[AppFuture, tuple[AppFuture, Dataset]]:
        if keep_trajectory:
            file = psiflow.context().new_file('data_', '.xyz')
        else:
            file = None
        result, output_future = self._propagate(
                model=model,
                keep_trajectory=keep_trajectory,
                file=file,
                )
        self.state_future   = unpack_i(result, 0)
        self.tag_future     = update_tag(self.tag_future, unpack_i(result, 1))
        self.counter_future = sum_counters(self.counter_future, unpack_i(result, 2))
        if hasattr(self, 'bias'):
            future = self.bias.evaluate(
                    Dataset([self.state_future]),
                    as_dataset=True)[0]
        else:
            future = copy_app_future(self.state_future) # necessary
        if reset_if_unsafe:
            self.reset(conditional=True)
        if keep_trajectory:
            assert output_future is not None
            return future, Dataset(None, data_future=output_future)
        else:
            return future

    def tag_unsafe(self) -> None:
        self.tag_future = copy_app_future('unsafe')

    def tag_safe(self) -> None:
        self.tag_future = copy_app_future('safe')

    def reset(self, conditional: bool = False) -> AppFuture:
        """Resets walker to a copy of its starting configuration

        If conditional is enabled, then the reset is only performed when the
        current tag of the walker is unsafe.

        """
        result = conditional_reset(
                self.state_future,
                self.start_future,
                self.tag_future,
                self.counter_future,
                conditional,
                )
        self.state_future   = unpack_i(result, 0)
        self.tag_future     = unpack_i(result, 1)
        self.counter_future = unpack_i(result, 2)
        return self.state_future

    def is_reset(self):
        return is_reset(self.counter_future)

    def set_state(self, atoms):
        self.state_future = app_reset_atoms(atoms)

    def set_start(self, atoms):
        self.start_future = app_reset_atoms(atoms)

    def copy(self) -> BaseWalker:
        walker = self.__class__(self.state_future, **self.parameters)
        walker.start_future = copy_app_future(self.start_future)
        walker.tag_future   = copy_app_future(self.tag_future)
        return walker

    def save(
            self,
            path: Union[Path, str],
            require_done: bool = True,
            ) -> tuple[DataFuture, DataFuture, DataFuture]:
        path = Path(path)
        assert path.is_dir()
        name = self.__class__.__name__
        path_start = path / 'start.xyz'
        path_state = path / 'state.xyz'
        path_pars  = path / (name + '.yaml')
        future_start = save_atoms(
                self.start_future,
                outputs=[File(str(path_start))],
                ).outputs[0]
        future_state = save_atoms(
                self.state_future,
                outputs=[File(str(path_state))],
                ).outputs[0]
        pars = self.parameters
        pars['counter'] = self.counter_future.result()
        future_pars = save_yaml(
                pars,
                outputs=[File(str(path_pars))],
                ).outputs[0]
        if require_done:
            future_start.result()
            future_state.result()
            future_pars.result()
        return future_start, future_state, future_pars

    @property
    def parameters(self) -> dict[str, Any]:
        """Returns dict of parameters to be passed into propagate"""
        return {'seed': self.seed}

    @classmethod
    def multiply(cls,
            nwalkers: int,
            data_start: Dataset,
            **kwargs) -> list[BaseWalker]:
        walkers = [cls(data_start[0], **kwargs) for i in range(nwalkers)]
        length = data_start.length().result()
        for i, walker in enumerate(walkers):
            walker.seed = i
            walker.set_start(data_start[i % length])
            walker.set_state(data_start[i % length])
        return walkers

    @classmethod
    def create_apps(cls) -> None:
        assert not (cls == BaseWalker) # should never be called directly
