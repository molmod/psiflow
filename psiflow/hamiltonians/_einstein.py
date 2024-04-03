from __future__ import annotations  # necessary for type-guarding class methods

from typing import Optional, Union

import numpy as np
import typeguard
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import FlowAtoms
from psiflow.hamiltonians.hamiltonian import Hamiltonian
from psiflow.hamiltonians.utils import check_forces, evaluate_function
from psiflow.utils import copy_app_future, dump_json


class EinsteinCalculator(Calculator):
    """ASE Calculator for a simple Einstein crystal"""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        centers: np.ndarray,
        force_constant: float,
        max_force: Optional[float] = None,
        **kwargs,
    ) -> None:
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.centers = centers
        self.force_constant = force_constant
        self.max_force = max_force

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        assert self.centers.shape[0] == len(atoms)
        forces = (-1.0) * self.force_constant * (atoms.get_positions() - self.centers)
        energy = (
            self.force_constant
            / 2
            * np.sum((atoms.get_positions() - self.centers) ** 2)
        )

        if self.max_force is not None:
            check_forces(forces, atoms, self.max_force)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }


@typeguard.typechecked
class EinsteinCrystal(Hamiltonian):
    """Dummy hamiltonian which represents a simple harmonic interaction"""

    def __init__(self, atoms: Union[FlowAtoms, AppFuture], force_constant: float):
        super().__init__()
        self.reference_geometry = copy_app_future(atoms)
        self.force_constant = force_constant
        self.input_files = []

        self.evaluate_app = python_app(evaluate_function, executors=["default_threads"])

    @property
    def parameters(self: Hamiltonian) -> dict:
        return {
            "reference_geometry": self.reference_geometry,
            "force_constant": self.force_constant,
        }

    @staticmethod
    def load_calculators(
        data: list[FlowAtoms],
        reference_geometry: Atoms,
        force_constant: float,
    ) -> tuple[list[EinsteinCalculator], np.ndarray]:
        import numpy as np

        from psiflow.hamiltonians._einstein import EinsteinCalculator

        assert sum([len(a) == len(reference_geometry) for a in data])
        assert sum([np.all(a.numbers == reference_geometry.numbers) for a in data])
        calculators = [
            EinsteinCalculator(reference_geometry.get_positions(), force_constant)
        ]
        index_mapping = np.zeros(len(data), dtype=int)
        return calculators, index_mapping

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if type(hamiltonian) is not EinsteinCrystal:
            return False
        if not np.allclose(self.force_constant, hamiltonian.force_constant):
            return False
        if self.reference_geometry.result() != hamiltonian.reference_geometry.result():
            return False
        return True

    def serialize(self) -> DataFuture:
        @python_app(executors=["default_threads"])
        def get_positions(atoms):
            return atoms.get_positions()

        return dump_json(
            hamiltonian=self.__class__.__name__,
            centers=get_positions(self.reference_geometry),
            force_constant=self.force_constant,
            outputs=[psiflow.context().new_file("hamiltonian_", ".json")],
        ).outputs[0]

    @staticmethod
    def deserialize(centers: list[list[float]], force_constant: float):
        return EinsteinCalculator(
            np.array(centers),
            force_constant,
        )
