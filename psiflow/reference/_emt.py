from __future__ import annotations  # necessary for type-guarding class methods

from typing import List, Union

import typeguard
from ase import Atoms
from parsl.app.app import python_app

import psiflow
from psiflow.data import FlowAtoms
from psiflow.reference.base import BaseReference


@typeguard.typechecked
def evaluate_emt(
    atoms: Union[Atoms, FlowAtoms],
    parameters: dict,
    inputs: List = [],
    outputs: List = [],
) -> FlowAtoms:
    from ase.calculators.emt import EMT

    if type(atoms) is not FlowAtoms:
        atoms = FlowAtoms.from_atoms(atoms)
    try:
        atoms.calc = EMT()
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        atoms.info["stress"] = atoms.get_stress(voigt=False)
        atoms.reference_status = True
    except NotImplementedError as e:
        atoms.reference_status = False
        print(e)
    atoms.calc = None
    atoms.reference_stderr = False
    atoms.reference_stdout = False
    return atoms


@typeguard.typechecked
class EMTReference(BaseReference):
    """Container class for EMT calculations (only used for testing purposes)"""

    @property
    def parameters(self):
        return {}

    @classmethod
    def create_apps(cls) -> None:
        app_evaluate_single = python_app(evaluate_emt, executors=["default_threads"])
        context = psiflow.context()
        context.register_app(cls, "evaluate_single", app_evaluate_single)
        # see https://stackoverflow.com/questions/1817183/using-super-with-a-class-method
        super(EMTReference, cls).create_apps()
