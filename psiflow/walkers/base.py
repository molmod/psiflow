from __future__ import annotations  # necessary for type-guarding class methods

from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import typeguard
from ase import Atoms
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, FlowAtoms, app_reset_atoms, write_atoms
from psiflow.models import BaseModel
from psiflow.utils import copy_app_future, save_yaml, sum_integers, unpack_i


@typeguard.typechecked
def _conditioned_reset(
    state: FlowAtoms,
    state0: FlowAtoms,
    counter: int,
    condition: bool,
) -> tuple[FlowAtoms, int]:
    from copy import deepcopy  # copy necessary!

    if condition:
        return deepcopy(state0), 0
    else:  # reset if unsafe
        return deepcopy(state), counter


conditioned_reset = python_app(_conditioned_reset, executors=["default_threads"])


@typeguard.typechecked
def _is_reset(counter: int) -> bool:
    return counter == 0


is_reset = python_app(_is_reset, executors=["default_threads"])


@typeguard.typechecked
class BaseWalker:
    def __init__(
        self,
        atoms: Union[Atoms, FlowAtoms, AppFuture],
        seed: int = 0,
    ) -> None:
        if type(atoms) is Atoms:
            atoms = FlowAtoms.from_atoms(atoms)
        self.state0 = app_reset_atoms(atoms)
        self.state = app_reset_atoms(atoms)
        self.counter = copy_app_future(0)  # counts nsteps

        self.seed = seed
        # apps for walkers are only created at the time when they are invoked,
        # as the execution definition of the model needs to be available.

    def _propagate(self, model):
        raise NotImplementedError

    def propagate(
        self,
        model: Optional[BaseModel] = None,
        keep_trajectory: bool = False,
    ) -> Union[NamedTuple, tuple[NamedTuple, Dataset]]:
        if keep_trajectory:
            file = psiflow.context().new_file("data_", ".xyz")
        else:
            file = None
        metadata, output_future = self._propagate(
            model=model,
            keep_trajectory=keep_trajectory,
            file=file,
        )
        self.state = metadata.state
        self.counter = sum_integers(
            self.counter,
            metadata.counter,
        )
        self.reset(metadata.reset)
        if keep_trajectory:
            assert output_future is not None
            return metadata, Dataset(None, data_future=output_future)
        else:
            return metadata

    def reset(self, condition: Union[None, bool, AppFuture] = None):
        """Resets walker to a copy of its starting configuration

        If condition is not None, it is assumed to be a boolean or an AppFuture representing a boolean
        which determines whether or not to reset.

        """
        if condition is None:
            condition = True
        result = conditioned_reset(
            self.state,
            self.state0,
            self.counter,
            condition,
        )
        self.state = unpack_i(result, 0)
        self.counter = unpack_i(result, 1)

    def is_reset(self):
        return is_reset(self.counter)

    def set_state(self, atoms):
        self.state = app_reset_atoms(atoms)

    def set_initial_state(self, atoms):
        self.state0 = app_reset_atoms(atoms)

    def copy(self) -> BaseWalker:
        walker = self.__class__(self.state, **self.parameters)
        walker.state0 = copy_app_future(self.state0)
        walker.counter = copy_app_future(self.counter)
        return walker

    def save(
        self,
        path: Union[Path, str],
        require_done: bool = True,
    ) -> tuple[DataFuture, DataFuture, DataFuture]:
        path = Path(path)
        path.mkdir(exist_ok=True)
        name = self.__class__.__name__
        path_state0 = path / "state0.xyz"
        path_state = path / "state.xyz"
        path_pars = path / (name + ".yaml")
        future_state0 = write_atoms(
            self.state0,
            outputs=[File(str(path_state0))],
        ).outputs[0]
        future_state = write_atoms(
            self.state,
            outputs=[File(str(path_state))],
        ).outputs[0]
        pars = self.parameters
        pars["counter"] = self.counter.result()
        future_pars = save_yaml(
            pars,
            outputs=[File(str(path_pars))],
        ).outputs[0]
        if require_done:
            future_state0.result()
            future_state.result()
            future_pars.result()
        return future_state0, future_state, future_pars

    @property
    def parameters(self) -> dict[str, Any]:
        """Returns dict of parameters to be passed into propagate"""
        return {"seed": self.seed}

    @classmethod
    def multiply(cls, nwalkers: int, data_start: Dataset, **kwargs) -> list[BaseWalker]:
        walkers = [cls(data_start[0], **kwargs) for i in range(nwalkers)]
        length = data_start.length().result()
        for i, walker in enumerate(walkers):
            walker.seed = i
            walker.set_initial_state(data_start[i % length])
            walker.set_state(data_start[i % length])
        return walkers

    @classmethod
    def create_apps(cls) -> None:
        assert not (cls == BaseWalker)  # should never be called directly
