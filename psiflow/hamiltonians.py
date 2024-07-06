from __future__ import annotations  # necessary for type-guarding class methods

from functools import partial
from pathlib import Path
from typing import Optional, Union, ClassVar

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.geometry import Geometry
from psiflow.data import Dataset, Computable, compute, aggregate_multiple
from psiflow.functions import ZeroFunction, EinsteinCrystalFunction, PlumedFunction, \
    HarmonicFunction, _apply
from psiflow.utils import copy_app_future, get_attribute, dump_json
from psiflow.tools.plumed import remove_comments_printflush


@typeguard.typechecked
@psiflow.serializable
class Hamiltonian(Computable):
    outputs: ClassVar[tuple] = ('energy', 'forces', 'stress')
    batch_size = 1000

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        raise NotImplementedError

    def __mul__(self, a: float) -> Hamiltonian:
        return MixtureHamiltonian([self], [a])

    def __add__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        if type(hamiltonian) is Zero:
            return self
        if type(hamiltonian) is MixtureHamiltonian:
            return hamiltonian.__add__(self)
        return 1.0 * self + 1.0 * hamiltonian

    def __sub__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        return self + (-1.0) * hamiltonian

    __rmul__ = __mul__  # handle float * Hamiltonian

    def serialize_function(self):
        return dump_json(
            function=self.function_name,
            outputs=[psiflow.context().new_file("hamiltonian_", ".json")],
            **self.parameters(),
        ).outputs[0]

    def parameters(self) -> dict:
        return {}


@typeguard.typechecked
@psiflow.serializable
class Zero(Hamiltonian):

    def __init__(self):
        apply_zero = python_app(_apply, executors=['default_threads'])
        self.app = partial(apply_zero, function_cls=ZeroFunction)

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if type(hamiltonian) is Zero:
            return True
        return False

    def __mul__(self, a: float) -> Hamiltonian:
        return Zero()

    def __add__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        return hamiltonian

    __rmul__ = __mul__  # handle float * Hamiltonian


@typeguard.typechecked
@psiflow.serializable
class MixtureHamiltonian(Hamiltonian):
    hamiltonians: list[Hamiltonian]
    coefficients: list[float]

    def __init__(
        self,
        hamiltonians: list[Hamiltonian],
        coefficients: list[float],
    ) -> None:
        self.hamiltonians = hamiltonians
        self.coefficients = coefficients

    def compute(  # override compute for efficient batching
        self,
        arg: Union[Dataset, AppFuture[list], list, AppFuture, Geometry],
        outputs: Union[str, list[str], None] = None,
        batch_size: Optional[int] = None,
    ) -> Union[list[AppFuture], AppFuture]:
        if outputs is None:
            outputs = list(self.__class__.outputs)
        apply_apps = [h.app for h in self.hamiltonians]
        reduce_func = partial(
            aggregate_multiple,
            coefficients=np.array(self.coefficients),
        )
        return compute(
            arg,
            *apply_apps,
            outputs_=outputs,
            reduce_func=reduce_func,
            batch_size=batch_size,
        )

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if type(hamiltonian) is not MixtureHamiltonian:
            return False
        if len(self.coefficients) != len(hamiltonian.coefficients):
            return False
        for i, h in enumerate(self.hamiltonians):
            if h not in hamiltonian.hamiltonians:
                return False
            coefficient = hamiltonian.coefficients[hamiltonian.hamiltonians.index(h)]
            if self.coefficients[i] != coefficient:
                return False
        return True

    def __mul__(self, a: float) -> Hamiltonian:
        return MixtureHamiltonian(
            self.hamiltonians,
            [c * a for c in self.coefficients],
        )

    def __len__(self) -> int:
        return len(self.coefficients)

    def __add__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        if type(hamiltonian) is Zero:
            return self
        if type(hamiltonian) is not MixtureHamiltonian:
            coefficients = list(self.coefficients)
            hamiltonians = list(self.hamiltonians)
            try:
                index = hamiltonians.index(hamiltonian)
                coefficients[index] += 1.0
            except ValueError:
                hamiltonians.append(hamiltonian)
                coefficients.append(1.0)
        else:
            coefficients = list(hamiltonian.coefficients)
            hamiltonians = list(hamiltonian.hamiltonians)
            for c, h in zip(self.coefficients, self.hamiltonians):
                try:
                    index = hamiltonians.index(h)
                    coefficients[index] += c
                except ValueError:
                    hamiltonians.append(h)
                    coefficients.append(c)
        return MixtureHamiltonian(hamiltonians, coefficients)

    def get_index(self, hamiltonian) -> Optional[int]:
        assert type(hamiltonian) is not MixtureHamiltonian
        if hamiltonian not in self.hamiltonians:
            return None
        return self.hamiltonians.index(hamiltonian)

    def get_indices(self, mixture) -> Optional[tuple[int, ...]]:
        assert type(mixture) is MixtureHamiltonian
        for h in mixture.hamiltonians:
            if h not in self.hamiltonians:
                return None
        indices = []
        for h in mixture.hamiltonians:
            indices.append(self.get_index(h))
        return tuple(indices)

    def get_coefficient(self, hamiltonian) -> Optional[float]:
        assert type(hamiltonian) is not MixtureHamiltonian
        if hamiltonian not in self.hamiltonians:
            return None
        return self.coefficients[self.hamiltonians.index(hamiltonian)]

    def get_coefficients(self, mixture) -> Optional[tuple[float, ...]]:
        assert type(mixture) is MixtureHamiltonian
        for h in mixture.hamiltonians:
            if h not in self.hamiltonians:
                return None
        coefficients = []
        for h in self.hamiltonians:
            coefficient = mixture.get_coefficient(h)
            if coefficient is None:
                coefficient = 0.0
            coefficients.append(coefficient)
        return tuple(coefficients)

    __rmul__ = __mul__  # handle float * Hamiltonian


@typeguard.typechecked
@psiflow.serializable
class EinsteinCrystal(Hamiltonian):
    reference_geometry: Union[Geometry, AppFuture]
    force_constant: float
    function_name: ClassVar[str] = 'EinsteinCrystalFunction'

    def __init__(
        self, geometry: Union[Geometry, AppFuture[Geometry]], force_constant: float
    ):
        super().__init__()
        self.reference_geometry = copy_app_future(geometry)
        self.force_constant = force_constant
        self.external = None  # needed
        self._create_apps()

    def _create_apps(self):
        apply_app = python_app(_apply, executors=['default_threads'])
        self.app = partial(
            apply_app,
            function_cls=EinsteinCrystalFunction,
            **self.parameters()
        )

    def parameters(self) -> dict:
        return {
            'force_constant': self.force_constant,
            'centers': get_attribute(self.reference_geometry, 'per_atom', 'positions'),
            'volume': get_attribute(self.reference_geometry, 'volume'),
        }

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if type(hamiltonian) is not EinsteinCrystal:
            return False
        if not np.allclose(self.force_constant, hamiltonian.force_constant):
            return False
        if self.reference_geometry != hamiltonian.reference_geometry:
            return False
        return True


@typeguard.typechecked
@psiflow.serializable
class PlumedHamiltonian(Hamiltonian):
    plumed_input: str
    external: Optional[psiflow._DataFuture]

    def __init__(
        self,
        plumed_input: str,
        external: Union[None, str, Path, File, DataFuture] = None,
    ):
        super().__init__()

        self.plumed_input = remove_comments_printflush(plumed_input)
        if type(external) in [str, Path]:
            external = File(str(external))
        if external is not None:
            assert external.filepath in self.plumed_input
        self.external = external
        self._create_apps()

    def _create_apps(self):
        apply_app = python_app(_apply, executors=['default_htex'])
        self.app = partial(
            apply_app,
            function_cls=PlumedFunction,
            **self.parameters(),
        )

    def parameters(self) -> dict:
        if self.external is not None:  # ensure parameters depends on self.external
            external = copy_app_future(self.external.filepath, inputs=[self.external])
        else:
            external = None
        return {
            'plumed_input': self.plumed_input,
            'external': external
        }

    def __eq__(self, other: Hamiltonian) -> bool:
        if type(other) is not type(self):
            return False
        if self.plumed_input != other.plumed_input:
            return False
        return True


@typeguard.typechecked
@psiflow.serializable
class Harmonic(Hamiltonian):
    reference_geometry: Union[Geometry, AppFuture[Geometry]]
    hessian: Union[np.ndarray, AppFuture[np.ndarray]]

    def __init__(
        self,
        reference_geometry: Union[Geometry, AppFuture[Geometry]],
        hessian: Union[np.ndarray, AppFuture[np.ndarray]],
    ):
        self.reference_geometry = reference_geometry
        self.hessian = hessian
        self._create_apps()

    def _create_apps(self):
        apply_app = python_app(_apply, executors=['default_threads'])
        self.app = partial(
            apply_app,
            function_cls=HarmonicFunction,
            **self.parameters(),
        )

    def parameters(self) -> dict:
        positions = get_attribute(self.reference_geometry, 'per_atom', 'positions')
        energy = get_attribute(self.reference_geometry, 'energy')
        return {
            'positions': positions,
            'energy': energy,
            'hessian': self.hessian,
        }

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


@typeguard.typechecked
@psiflow.serializable
class MACEHamiltonian(Hamiltonian):
    pass


def get_mace_mp0():
    pass