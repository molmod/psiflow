"""
TODO: these imports are outdated.. Is this module still used?
"""
from __future__ import annotations  # necessary for type-guarding class methods

from functools import partial
from typing import Optional, Union

import typeguard
from ase.units import kJ, mol
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, batch_apply
from psiflow.geometry import Geometry
from psiflow.hamiltonians._plumed import PlumedHamiltonian
from psiflow.hamiltonians.hamiltonian import Hamiltonian


@typeguard.typechecked
def insert_in_state(
    state: Geometry,
    name: str,
) -> Geometry:
    value = state.energy
    state.order[name] = value
    state.energy = None
    return state


@typeguard.typechecked
def _insert(
    state_or_states: Union[Geometry, list[Geometry]],
    name: str,
) -> Union[list[Geometry], Geometry]:
    if not isinstance(state_or_states, list):
        return insert_in_state(state_or_states, name)
    else:
        for state in state_or_states:
            insert_in_state(state, name)  # modify list in place
        return state_or_states


insert = python_app(_insert, executors=["default_threads"])


@typeguard.typechecked
def insert_in_dataset(
    data: Dataset,
    name: str,
) -> Dataset:
    geometries = insert(
        data.geometries(),
        name,
    )
    return Dataset(geometries)


@typeguard.typechecked
class OrderParameter:
    # TODO: batched evaluation

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, state: Union[Geometry, AppFuture]) -> AppFuture:
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


@typeguard.typechecked
@psiflow.serializable
class HamiltonianOrderParameter(OrderParameter):
    name: str
    hamiltonian: Hamiltonian

    def __init__(self, name: str, hamiltonian: Hamiltonian):
        super().__init__(name)
        self.hamiltonian = hamiltonian

    def evaluate(
        self,
        arg: Union[Dataset, Geometry, AppFuture[Geometry]],
        batch_size: Optional[int] = 100,
    ) -> Union[Dataset, AppFuture]:
        if isinstance(arg, Dataset):
            # avoid batching the dataset twice:
            # apply hamiltonian in batched sense and put insert afterwards
            funcs = [
                self.hamiltonian.single_evaluate,
                partial(insert_in_dataset, name=self.name),
            ]
            future = batch_apply(
                funcs,
                batch_size,
                arg.length(),
                inputs=[arg.extxyz],
                outputs=[psiflow.context().new_file("data_", ".xyz")],
            )
            return Dataset(None, future.outputs[0])
        else:
            state = self.hamiltonian.evaluate(arg)
            return insert(state, self.name)

    def __eq__(self, other):
        if type(other) is not HamiltonianOrderParameter:
            return False
        return self.hamiltonian == other.hamiltonian

    @classmethod
    def from_plumed(
        cls, name: str, hamiltonian: PlumedHamiltonian
    ) -> HamiltonianOrderParameter:
        assert name in hamiltonian.plumed_input()
        action_prefixes = [
            "ABMD",
            "BIASVALUE",
            "EXTENDED_LAGRANGIAN",
            "EXTERNAL",
            "LOWER_WALLS",
            "MAXENT",
            "METAD",
            "MOVINGRESTRAINT",
            "PBMETAD",
            "RESTRAINT",
            "UPPER_WALLS",
            "RESTART",
        ]
        lines = hamiltonian.plumed_input().split("\n")
        new_lines = []
        for line in lines:
            found = [p in line for p in action_prefixes]
            if sum(found, start=False):
                continue
            else:
                new_lines.append(line)
        ev_to_kjmol = 1 / (
            kJ / mol
        )  # compensate plumed to ASE unit conversion of 'energy'
        new_lines.append(
            "rescaled: MATHEVAL ARG={} FUNC=x*{} PERIODIC=NO".format(name, ev_to_kjmol)
        )
        new_lines.append("BIASVALUE ARG=rescaled")
        return HamiltonianOrderParameter(
            name=name,
            hamiltonian=PlumedHamiltonian(plumed_input="\n".join(new_lines)),
        )
