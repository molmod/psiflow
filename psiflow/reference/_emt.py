from __future__ import annotations  # necessary for type-guarding class methods

from typing import Union

import typeguard
from parsl.app.app import python_app

import psiflow
from psiflow.geometry import Geometry, NullState
from psiflow.reference.reference import Reference


@typeguard.typechecked
def evaluate_emt(
    geometry: Geometry,
    inputs: list = [],
    outputs: list = [],
) -> Geometry:
    from ase import Atoms
    from ase.calculators.emt import EMT

    atoms = Atoms(
        numbers=geometry.per_atom.numbers[:],
        positions=geometry.per_atom.positions[:],
        cell=geometry.cell,
        pbc=geometry.periodic,
    )
    try:
        atoms.calc = EMT()
        geometry.energy = atoms.get_potential_energy()
        geometry.per_atom.forces[:] = atoms.get_forces()
        geometry.stress = atoms.get_stress(voigt=False)
    except NotImplementedError as e:
        print(e)
        geometry = NullState
    return geometry


@typeguard.typechecked
@psiflow.serializable
class EMT(Reference):
    properties: list[str]

    def __init__(self, properties: Union[list, tuple] = ("energy", "forces")):
        self.properties = list(properties)
        self._create_apps()

    def _create_apps(self):
        self.evaluate_single = python_app(evaluate_emt, executors=["default_threads"])
