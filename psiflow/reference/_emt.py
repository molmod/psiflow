from __future__ import annotations  # necessary for type-guarding class methods

import typeguard
from parsl.app.app import python_app

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
class EMT(Reference):
    """Container class for EMT calculations (only used for testing purposes)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluate_single = python_app(evaluate_emt, executors=["default_threads"])
