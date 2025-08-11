from __future__ import annotations  # necessary for type-guarding class methods

import urllib
from functools import partial
from pathlib import Path
from typing import ClassVar, Optional, Union, Callable

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Computable, Dataset, aggregate_multiple, compute
from psiflow.functions import (
    EinsteinCrystalFunction,
    HarmonicFunction,
    MACEFunction,
    PlumedFunction,
    ZeroFunction,
    DispersionFunction,
    _apply,
)
from psiflow.geometry import Geometry
from psiflow.utils._plumed import remove_comments_printflush
from psiflow.utils.apps import copy_app_future, get_attribute
from psiflow.utils.io import dump_json


@typeguard.typechecked
@psiflow.serializable
class Hamiltonian(Computable):
    # TODO: app is actually an instance variable, but serialization complains..
    outputs: ClassVar[tuple] = ("energy", "forces", "stress")
    batch_size = 1000
    app: ClassVar[Callable]
    function_name: ClassVar[str]

    def compute(
        self,
        arg: Union[Dataset, AppFuture[list], list, AppFuture, Geometry],
        *outputs: Optional[str],
        batch_size: Optional[int] = -1,  # if -1: take class default TODO: why?
    ) -> Union[list[AppFuture], AppFuture]:
        if len(outputs) == 0:
            outputs = tuple(self.__class__.outputs)
        if batch_size == -1:
            batch_size = self.__class__.batch_size
        return compute(
            arg,
            self.app,
            outputs_=outputs,
            batch_size=batch_size,
        )

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        raise NotImplementedError

    def __mul__(self, a: float) -> MixtureHamiltonian:
        return MixtureHamiltonian([self], [a])

    def __add__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        if type(hamiltonian) is Zero:
            return self
        mixture = MixtureHamiltonian([self], [1.0])
        return mixture.__add__(hamiltonian)

    def __sub__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        return self + (-1.0) * hamiltonian

    __rmul__ = __mul__  # handle float * Hamiltonian

    def serialize_function(self, **kwargs):
        parameters = self.parameters()
        for key, value in kwargs.items():
            if key in parameters:
                parameters[key] = value
        return dump_json(
            function_name=self.function_name,
            outputs=[psiflow.context().new_file("hamiltonian_", ".json")],
            **parameters,
        ).outputs[0]

    def parameters(self) -> dict:
        return {}


@typeguard.typechecked
@psiflow.serializable
class Zero(Hamiltonian):

    def __init__(self):
        apply_zero = python_app(_apply, executors=["default_threads"])
        self.app = partial(apply_zero, function_cls=ZeroFunction)

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if type(hamiltonian) is Zero:
            return True
        return False

    def __mul__(self, a: float) -> Hamiltonian:
        return Zero()

    def __add__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        # (Zero + Hamiltonian) is different from (Hamiltonian + Zero)
        return hamiltonian

    __rmul__ = __mul__  # handle float * Zero


@typeguard.typechecked
@psiflow.serializable
class MixtureHamiltonian(Hamiltonian):
    hamiltonians: list[Hamiltonian]
    coefficients: list[float]

    def __init__(
        self,
        hamiltonians: Union[tuple, list][Hamiltonian],
        coefficients: Union[tuple, list][float],
    ) -> None:
        assert len(hamiltonians) == len(coefficients)
        self.hamiltonians = list(hamiltonians)
        self.coefficients = list(coefficients)

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

    def __eq__(self, hamiltonian: Hamiltonian | MixtureHamiltonian) -> bool:
        if type(hamiltonian) is not MixtureHamiltonian:
            return False
        if len(self.coefficients) != len(hamiltonian.coefficients):
            return False
        for c, h in zip(self.coefficients, self.hamiltonians):
            if (idx := hamiltonian.get_index(h)) is None:
                return False
            if c != hamiltonian.coefficients[idx]:
                return False
        return True

    def __mul__(self, a: float) -> MixtureHamiltonian:
        return MixtureHamiltonian(
            self.hamiltonians,
            [c * a for c in self.coefficients],
        )

    __rmul__ = __mul__  # handle float * MixtureHamiltonian

    def __len__(self) -> int:
        return len(self.coefficients)

    def __add__(self, hamiltonian: Hamiltonian) -> MixtureHamiltonian:
        if type(hamiltonian) is Zero:
            return self
        if type(hamiltonian) is not MixtureHamiltonian:
            hamiltonian = 1.0 * hamiltonian  # turn into mixture

        coefficients = list(hamiltonian.coefficients)
        hamiltonians = list(hamiltonian.hamiltonians)
        for c, h in zip(self.coefficients, self.hamiltonians):
            if (idx := hamiltonian.get_index(h)) is None:
                hamiltonians.append(h)
                coefficients.append(c)
            else:
                coefficients[idx] += c
        return MixtureHamiltonian(hamiltonians, coefficients)

    def get_index(self, hamiltonian: Hamiltonian) -> Optional[int]:
        assert type(hamiltonian) is not MixtureHamiltonian
        try:
            return self.hamiltonians.index(hamiltonian)
        except ValueError:
            return None

    def get_indices(self, mixture: MixtureHamiltonian) -> Optional[tuple[int, ...]]:
        # TODO: why do we not just return None for the missing components?
        assert type(mixture) is MixtureHamiltonian
        indices = []
        for h in mixture.hamiltonians:
            indices.append(self.get_index(h))
        if any([idx is None for idx in indices]):
            return None
        return tuple(indices)

    def get_coefficient(self, hamiltonian: Hamiltonian) -> Optional[float]:
        assert type(hamiltonian) is not MixtureHamiltonian
        if (idx := self.get_index(hamiltonian)) is None:
            return None
        return self.coefficients[idx]

    def get_coefficients(
        self, mixture: MixtureHamiltonian
    ) -> Optional[tuple[float, ...]]:
        assert type(mixture) is MixtureHamiltonian
        for h in mixture.hamiltonians:
            # every mixture component must be a component of self
            if h not in self.hamiltonians:
                return None

        # components of self that are not in mixture get coefficient 0
        coefficients = [(mixture.get_coefficient(h) or 0) for h in self.hamiltonians]
        return tuple(coefficients)

    def get_named_components(self) -> list[str]:
        """Create unique string name for every hamiltonian component to be used in i-Pi sampling"""
        names, counts = [], {}
        for h in self.hamiltonians:
            name = h.__class__.__name__
            counts.setdefault(name, 0)
            names.append(f"{name}{counts[name]}")
            counts[name] += 1
        return names

    def serialize(self, **kwargs) -> list[DataFuture]:
        return [h.serialize_function(**kwargs) for h in self.hamiltonians]


@typeguard.typechecked
@psiflow.serializable
class EinsteinCrystal(Hamiltonian):
    reference_geometry: Union[Geometry, AppFuture]
    force_constant: float
    function_name: ClassVar[str] = "EinsteinCrystalFunction"

    def __init__(
        self, geometry: Union[Geometry, AppFuture[Geometry]], force_constant: float
    ):
        super().__init__()
        self.reference_geometry = copy_app_future(geometry)
        self.force_constant = force_constant
        self.external = None  # needed
        self._create_apps()

    def _create_apps(self):
        apply_app = python_app(_apply, executors=["default_threads"])
        self.app = partial(
            apply_app, function_cls=EinsteinCrystalFunction, **self.parameters()
        )

    def parameters(self) -> dict:
        return {
            "force_constant": self.force_constant,
            "centers": get_attribute(self.reference_geometry, "per_atom", "positions"),
            "volume": get_attribute(self.reference_geometry, "volume"),
        }

    def __eq__(self, hamiltonian: Hamiltonian | EinsteinCrystal) -> bool:
        if (
            not isinstance(hamiltonian, EinsteinCrystal)
            or not np.allclose(self.force_constant, hamiltonian.force_constant)
            or self.reference_geometry != hamiltonian.reference_geometry
        ):
            return False
        return True


@typeguard.typechecked
@psiflow.serializable
class PlumedHamiltonian(Hamiltonian):
    plumed_input: str
    external: Optional[psiflow._DataFuture]
    function_name: ClassVar[str] = "PlumedFunction"

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
        apply_app = python_app(_apply, executors=["default_htex"])
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
        return {"plumed_input": self.plumed_input, "external": external}

    def __eq__(self, other: Hamiltonian | PlumedHamiltonian) -> bool:
        if (
            not isinstance(other, PlumedHamiltonian)
            or self.plumed_input != other.plumed_input
        ):
            return False
        return True


@typeguard.typechecked
@psiflow.serializable
class Harmonic(Hamiltonian):
    reference_geometry: Union[Geometry, AppFuture[Geometry]]
    hessian: Union[np.ndarray, AppFuture[np.ndarray]]
    function_name: ClassVar[str] = "HarmonicFunction"

    def __init__(
        self,
        reference_geometry: Union[Geometry, AppFuture[Geometry]],
        hessian: Union[np.ndarray, AppFuture[np.ndarray]],
    ):
        # TODO: why not copy_app_future(geometry) like others?
        self.reference_geometry = reference_geometry
        self.hessian = hessian
        self._create_apps()

    def _create_apps(self):
        apply_app = python_app(_apply, executors=["default_threads"])
        self.app = partial(
            apply_app,
            function_cls=HarmonicFunction,
            **self.parameters(),
        )

    def parameters(self) -> dict:
        positions = get_attribute(self.reference_geometry, "per_atom", "positions")
        energy = get_attribute(self.reference_geometry, "energy")
        return {
            "positions": positions,
            "energy": energy,
            "hessian": self.hessian,
        }

    def __eq__(self, hamiltonian: Hamiltonian | Harmonic) -> bool:
        if type(hamiltonian) is not Harmonic:
            return False
        if hamiltonian.reference_geometry != self.reference_geometry:
            return False

        # TODO: why this check? Is it not always an ndarray?
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
class D3Hamiltonian(Hamiltonian):
    method: str
    damping: str
    function_name: ClassVar[str] = "DispersionFunction"

    def __init__(self, method: str, damping: str = "d3bj"):
        self.method, self.damping = method, damping
        self._create_apps()

    def _create_apps(self):
        # TODO: does this make sense? GPU settings are useless for example
        evaluation = psiflow.context().definitions["ModelEvaluation"]
        apply_app = python_app(_apply, executors=["ModelEvaluation"])
        resources = evaluation.wq_resources(1)
        resources.pop("gpus", None)

        # execution-side parameters of function are not included in self.parameters()
        self.app = partial(
            apply_app,
            function_cls=DispersionFunction,
            parsl_resource_specification=resources,
            **self.parameters(),
            num_threads=resources["cores"],
        )

    def parameters(self) -> dict:
        return {"method": self.method, "damping": self.damping}

    def __eq__(self, hamiltonian: Hamiltonian | D3Hamiltonian) -> bool:
        if (
            not isinstance(hamiltonian, D3Hamiltonian)
            or self.method != hamiltonian.method
            or self.damping != hamiltonian.damping
        ):
            return False
        return True


@typeguard.typechecked
@psiflow.serializable
class MACEHamiltonian(Hamiltonian):
    external: psiflow._DataFuture
    atomic_energies: dict[str, float]
    function_name: ClassVar[str] = "MACEFunction"

    def __init__(
        self,
        external: Union[Path, str, psiflow._DataFuture],
        atomic_energies: dict[str, float],
    ):
        self.atomic_energies = atomic_energies
        if type(external) in [str, Path]:
            self.external = File(external)
        else:
            self.external = external
        self._create_apps()

    def _create_apps(self):
        evaluation = psiflow.context().definitions["ModelEvaluation"]
        apply_app = python_app(_apply, executors=["ModelEvaluation"])
        resources = evaluation.wq_resources(1)

        # execution-side parameters of function are not included in self.parameters()
        self.app = partial(
            apply_app,
            function_cls=MACEFunction,
            parsl_resource_specification=resources,
            **self.parameters(),
        )

    def parameters(self) -> dict:
        model_path = copy_app_future(self.external.filepath, inputs=[self.external])
        evaluation = psiflow.context().definitions["ModelEvaluation"]
        return {
            "model_path": model_path,
            "atomic_energies": self.atomic_energies,
            "ncores": evaluation.cores_per_worker,
            "dtype": "float32",
            "device": "gpu" if evaluation.gpu else "cpu",
            "env_vars": evaluation.env_vars,
        }

    def __eq__(self, hamiltonian: Hamiltonian | MACEHamiltonian) -> bool:
        if type(hamiltonian) is not MACEHamiltonian:
            return False
        if self.external.filepath != hamiltonian.external.filepath:
            return False
        if len(self.atomic_energies) != len(hamiltonian.atomic_energies):
            return False
        for symbol, energy in self.atomic_energies.items():
            if not np.allclose(
                energy,
                hamiltonian.atomic_energies[symbol],
            ):
                return False
        return True

    # TODO: the methods below are outdated..

    @classmethod
    def mace_mp0(cls, size: str = "small") -> MACEHamiltonian:
        urls = dict(
            small="https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",  # 2023-12-10-mace-128-L0_energy_epoch-249.model
            large="https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
        )
        assert size in urls
        parsl_file = psiflow.context().new_file("mace_mp_", ".pth")
        urllib.request.urlretrieve(
            urls[size],
            parsl_file.filepath,
        )
        return cls(parsl_file, {})

    @classmethod
    def mace_cc(cls) -> MACEHamiltonian:
        url = "https://github.com/molmod/psiflow/raw/main/examples/data/ani500k_cc_cpu.model"
        parsl_file = psiflow.context().new_file("mace_mp_", ".pth")
        urllib.request.urlretrieve(
            url,
            parsl_file.filepath,
        )
        return cls(parsl_file, {})
