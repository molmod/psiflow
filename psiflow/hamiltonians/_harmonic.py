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
from psiflow.data import Geometry
from psiflow.hamiltonians.hamiltonian import Hamiltonian
from psiflow.hamiltonians.utils import evaluate_function
from psiflow.utils import dump_json


@typeguard.typechecked
class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        positions: np.ndarray,
        hessian: np.ndarray,
        energy: float,
        max_force: Optional[float] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        assert hessian.shape[0] == 3 * positions.shape[0]
        self.positions = positions
        self.hessian = hessian
        self.energy = energy
        self.max_force = max_force

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)
        assert self.hessian.shape[0] == 3 * len(atoms)

        pos = atoms.positions.reshape(-1)

        delta = pos - self.positions.reshape(-1)
        grad = np.dot(self.hessian, delta)
        energy = self.energy + 0.5 * np.dot(delta, grad)

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": (-1.0) * grad.reshape(-1, 3),
        }
        if sum(atoms.pbc):
            self.results["stress"] = full_3x3_to_voigt_6_stress(np.zeros((3, 3)))


@typeguard.typechecked
class Harmonic(Hamiltonian):
    reference_geometry: Union[Geometry, AppFuture]
    hessian: Union[np.ndarray, AppFuture]

    def __init__(
        self,
        reference_geometry: Union[Geometry, AppFuture],
        hessian: Union[np.ndarray, AppFuture],
    ):
        self.reference_geometry = reference_geometry
        self.hessian = hessian
        self.external = None  # needed
        self._create_apps()

    def _create_apps(self):
        self.evaluate_app = python_app(evaluate_function, executors=["default_threads"])

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if type(hamiltonian) is not Harmonic:
            return False
        if hamiltonian.reference_geometry != self.reference_geometry:
            return False

        # slightly different check for numpy arrays
        is_array0 = type(hamiltonian.hessian) is np.ndarray
        is_array1 = type(self.hessian) is np.ndarray
        if is_array0 and is_array1:
            equal = np.allclose(
                hamiltonian.hessian,
                self.hessian,
            )
        else:
            equal = hamiltonian.hessian == self.hessian

        if not equal:
            return False
        return True

    def serialize_calculator(self) -> DataFuture:
        @python_app(executors=["default_threads"])
        def get_positions(geometry: Geometry):
            return geometry.per_atom.positions.copy().astype(float)

        @python_app(executors=["default_threads"])
        def get_energy(geometry: Geometry):
            return geometry.energy

        return dump_json(
            hamiltonian=self.__class__.__name__,
            positions=get_positions(self.reference_geometry),
            hessian=self.hessian,
            energy=get_energy(self.reference_geometry),
            outputs=[psiflow.context().new_file("hamiltonian_", ".json")],
        ).outputs[0]

    @staticmethod
    def deserialize_calculator(
        positions: list[list[float]],
        hessian: list[list[float]],
        energy: float,
    ):
        return HarmonicCalculator(
            np.array(positions),
            np.array(hessian),
            energy,
        )

    @property
    def parameters(self: Hamiltonian) -> dict:
        return {
            "hessian": self.hessian,
            "reference_geometry": self.reference_geometry,
        }

    @staticmethod
    def load_calculators(
        data: list[Geometry],
        external: Optional[File],
        hessian: np.ndarray,
        reference_geometry: Geometry,
    ) -> tuple[list[HarmonicCalculator], np.ndarray]:
        from psiflow.hamiltonians._harmonic import HarmonicCalculator

        assert sum([len(g) == len(reference_geometry) for g in data])
        numbers = reference_geometry.per_atom.numbers
        assert sum([np.all(g.per_atom.numbers == numbers) for g in data])

        harmonic = HarmonicCalculator(
            reference_geometry.per_atom.positions,
            hessian,
            reference_geometry.energy,
        )
        calculators = [harmonic]
        index_mapping = np.zeros(len(data), dtype=int)
        return calculators, index_mapping
