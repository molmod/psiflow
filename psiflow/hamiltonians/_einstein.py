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
from psiflow.utils import copy_app_future, dump_json


@typeguard.typechecked
@psiflow.serializable
class EinsteinCrystal(Hamiltonian):
    reference_geometry: Union[Geometry, AppFuture]
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
        self.evaluate_app = python_app(evaluate_function, executors=["default_htex"])

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
    ) -> tuple[list[Any], np.ndarray]:
        import numpy as np

        from psiflow.geometry import NullState
        from psiflow.hamiltonians.utils import EinsteinCalculator

        natoms = len(reference_geometry)
        numbers = reference_geometry.per_atom.numbers
        for g in data:
            if g == NullState:
                continue
            assert len(g) == natoms
            assert np.all(g.per_atom.numbers == numbers)

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
        if self.reference_geometry != hamiltonian.reference_geometry:
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
        from psiflow.hamiltonians.utils import EinsteinCalculator

        return EinsteinCalculator(
            np.array(centers),
            force_constant,
            volume,
        )
