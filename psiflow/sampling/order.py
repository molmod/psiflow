from __future__ import annotations  # necessary for type-guarding class methods

from typing import Union

import typeguard
from ase.units import kJ, mol
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, Geometry
from psiflow.hamiltonians._plumed import PlumedHamiltonian
from psiflow.hamiltonians.hamiltonian import Hamiltonian


def _insert_in_state(
    state: Geometry,
    name: str,
) -> Geometry:
    value = state.energy
    state.order[name] = value
    state.energy = None
    return state


insert_in_state = python_app(_insert_in_state, executors=["default_threads"])


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

    def evaluate(self, state: Union[Geometry, AppFuture]) -> AppFuture:
        return insert_in_state(
            self.hamiltonian.evaluate(Dataset([state]))[0],
            self.name,
        )

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
