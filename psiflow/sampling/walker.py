from __future__ import annotations  # necessary for type-guarding class methods

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms, check_equality
from psiflow.hamiltonians.hamiltonian import Hamiltonian
from psiflow.sampling.output import SimulationOutput
from psiflow.utils import copy_app_future


@typeguard.typechecked
def _update_walker(
    state: FlowAtoms,
    status: int,
    start: FlowAtoms,
) -> FlowAtoms:
    # success or timeout are OK
    if status in [0, 1]:
        return state
    else:
        return start


update_walker = python_app(_update_walker, executors=["default_threads"])


@typeguard.typechecked
def _conditioned_reset(
    condition: bool,
    state: FlowAtoms,
    start: FlowAtoms,
) -> FlowAtoms:
    from copy import deepcopy  # copy necessary!

    if condition:
        return deepcopy(start)
    else:
        return deepcopy(state)


conditioned_reset = python_app(_conditioned_reset, executors=["default_threads"])


@dataclass
@typeguard.typechecked
class Walker:
    start: Union[FlowAtoms, AppFuture]
    hamiltonian: Hamiltonian
    state: Union[FlowAtoms, AppFuture, None] = None
    temperature: Optional[float] = 300
    pressure: Optional[float] = None
    nbeads: int = 1
    replica_exchange: int = -1
    periodic: Union[bool, AppFuture] = True

    def __post_init__(self):
        if self.state is None:
            self.state = copy_app_future(self.start)
        if type(self.start) is AppFuture:
            start = self.start.result()  # blocking
        else:
            start = self.start
        self.periodic = np.all(start.pbc)

    def reset(self, condition: Union[AppFuture, bool] = True):
        self.state = conditioned_reset(condition, self.state, self.start)

    def is_reset(self) -> AppFuture:
        return check_equality(self.start, self.state)

    def update(self, output: SimulationOutput) -> None:
        self.state = update_walker(
            output.state,
            output.status,
            self.start,
        )

    @staticmethod
    def is_similar(w0: Walker, w1: Walker):
        similar_T = (w0.temperature is None) == (w1.temperature is None)
        similar_P = (w0.pressure is None) == (w1.pressure is None)
        similar_pimd = w0.pimd == w1.pimd
        similar_rex = w0.replica_exchange == w1.replica_exchange
        similar_pbc = w0.periodic == w1.periodic
        return similar_T and similar_P and similar_pimd and similar_rex and similar_pbc

    @property
    def pimd(self):
        return self.nbeads != 1

    @property
    def nve(self):
        return (self.temperature is None) and (self.pressure is None)

    @property
    def nvt(self):
        return (self.temperature is not None) and (self.pressure is None)

    @property
    def npt(self):
        return (self.temperature is not None) and (self.pressure is not None)


@typeguard.typechecked
def partition(walkers: list[Walker]) -> list[list[Walker]]:
    partitions = []
    for walker in walkers:
        found = False
        for partition in partitions:
            if Walker.is_similar(walker, partition[0]):
                partition.append(walker)
                found = True
        if not found:
            partitions.append([walker])
    return partitions
