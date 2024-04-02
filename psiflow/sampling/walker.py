from __future__ import annotations  # necessary for type-guarding class methods

from dataclasses import dataclass
from typing import Optional, Union

import typeguard
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms
from psiflow.hamiltonians.hamiltonian import Hamiltonian


@dataclass
@typeguard.typechecked
class Walker:
    start: Union[FlowAtoms, AppFuture]
    state: Union[FlowAtoms, AppFuture]
    hamiltonian: Hamiltonian
    temperature: Optional[float] = 300
    pressure: Optional[float] = None
    nbeads: int = 1
    replica_exchange: int = -1

    def reset(self):
        pass

    def is_reset(self) -> bool:
        pass

    @staticmethod
    def is_similar(w0: Walker, w1: Walker):
        similar_T = (w0.temperature is None) == (w1.temperature is None)
        similar_P = (w0.pressure is None) == (w1.pressure is None)
        similar_pimd = w0.pimd == w1.pimd
        return similar_T and similar_P and similar_pimd

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
