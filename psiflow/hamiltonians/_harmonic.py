from __future__ import annotations  # necessary for type-guarding class methods

from typing import Any, Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.hamiltonians.hamiltonian import Hamiltonian, evaluate_function
from psiflow.utils import dump_json


@typeguard.typechecked
@psiflow.serializable
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
        self.evaluate_app = python_app(evaluate_function, executors=["default_htex"])

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
        from psiflow.hamiltonians.utils import HarmonicCalculator

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
    ) -> tuple[list[Any], np.ndarray]:
        from psiflow.geometry import NullState
        from psiflow.hamiltonians.utils import HarmonicCalculator

        natoms = len(reference_geometry)
        numbers = reference_geometry.per_atom.numbers
        for g in data:
            if g == NullState:
                continue
            assert len(g) == natoms
            assert np.all(g.per_atom.numbers == numbers)

        harmonic = HarmonicCalculator(
            reference_geometry.per_atom.positions,
            hessian,
            reference_geometry.energy,
        )
        calculators = [harmonic]
        index_mapping = np.zeros(len(data), dtype=int)
        return calculators, index_mapping
