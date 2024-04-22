from __future__ import annotations  # necessary for type-guarding class methods

from typing import Optional, Union

import numpy as np
import typeguard
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.hamiltonians.hamiltonian import Hamiltonian
from psiflow.hamiltonians.utils import check_forces, evaluate_function
from psiflow.utils import copy_app_future, dump_json


class EinsteinCalculator(Calculator):
    """ASE Calculator for a simple Einstein crystal"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        centers: np.ndarray,
        force_constant: float,
        volume: float,
        max_force: Optional[float] = None,
        **kwargs,
    ) -> None:
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.centers = centers
        self.force_constant = force_constant
        self.volume = volume
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
        if sum(atoms.pbc) and self.volume > 0.0:
            delta = np.linalg.det(atoms.cell) - self.volume
            self.results["stress"] = self.force_constant * np.eye(3) * delta
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])


@typeguard.typechecked
@psiflow.serializable
class EinsteinCrystal(Hamiltonian):
    geometry: Union[Geometry, AppFuture]
    force_constant: float

    def __init__(
        self, geometry: Union[Geometry, AppFuture[Geometry]], force_constant: float
    ):
        super().__init__()
        self.reference_geometry = copy_app_future(geometry)
        self.force_constant = force_constant
        self.external = None  # needed
        self._create_apps()

    def _create_apps(self):
        self.evaluate_app = python_app(evaluate_function, executors=["default_threads"])

    @property
    def parameters(self: Hamiltonian) -> dict:
        return {
            "reference_geometry": self.reference_geometry,
            "force_constant": self.force_constant,
        }

    @staticmethod
    def load_calculators(
        data: list[Geometry],
        external: Optional[File],
        reference_geometry: Geometry,
        force_constant: float,
    ) -> tuple[list[EinsteinCalculator], np.ndarray]:
        import numpy as np

        from psiflow.hamiltonians._einstein import EinsteinCalculator

        assert sum([len(g) == len(reference_geometry) for g in data])
        numbers = reference_geometry.per_atom.numbers
        assert sum([np.all(g.per_atom.numbers == numbers) for g in data])

        einstein = EinsteinCalculator(
            reference_geometry.per_atom.positions,
            force_constant,
            np.linalg.det(reference_geometry.cell),
        )
        calculators = [einstein]
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

    def serialize_calculator(self) -> DataFuture:
        @python_app(executors=["default_threads"])
        def get_positions(geometry: Geometry):
            return geometry.per_atom.positions.copy().astype(float)

        @python_app(executors=["default_threads"])
        def get_volume(geometry: Geometry):
            return float(np.linalg.det(geometry.cell))

        return dump_json(
            hamiltonian=self.__class__.__name__,
            centers=get_positions(self.reference_geometry),
            force_constant=self.force_constant,
            volume=get_volume(self.reference_geometry),
            outputs=[psiflow.context().new_file("hamiltonian_", ".json")],
        ).outputs[0]

    @staticmethod
    def deserialize_calculator(
        centers: list[list[float]],
        force_constant: float,
        volume: float,
    ):
        return EinsteinCalculator(
            np.array(centers),
            force_constant,
            volume,
        )
