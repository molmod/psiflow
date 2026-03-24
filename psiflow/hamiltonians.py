from functools import partial
from pathlib import Path
from typing import Optional, Union, Callable, Sequence, Any, ClassVar
from dataclasses import dataclass, field, InitVar

import numpy as np
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture, Future

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
from psiflow.utils.apps import get_attribute
from psiflow.utils.io import dump_json


# TODO: comparison logic in __eq__ only works for hamiltonians without futures
# TODO: dataclasses automatically generate __eq__


apply_threads = python_app(_apply, executors=["default_threads"])
apply_htex = python_app(_apply, executors=["default_htex"])
apply_modelevaluation = python_app(_apply, executors=["ModelEvaluation"])


# TODO: why have the Computable class?
class Hamiltonian(Computable):
    app: Callable
    batch_size = 1000
    outputs: ClassVar[tuple] = ("energy", "forces", "stress")
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
        return compute(arg, self.get_app(), outputs_=outputs, batch_size=batch_size)

    def __eq__(self, hamiltonian: "Hamiltonian") -> bool:
        raise NotImplementedError

    def __mul__(self, a: float) -> "MixtureHamiltonian":
        return MixtureHamiltonian([self], [a])

    def __add__(self, hamiltonian: "Hamiltonian") -> "MixtureHamiltonian | Hamiltonian":
        if type(hamiltonian) is Zero:
            return self
        mixture = MixtureHamiltonian([self], [1.0])
        return mixture.__add__(hamiltonian)

    def __sub__(self, hamiltonian: "Hamiltonian") -> "MixtureHamiltonian":
        return self + hamiltonian * (-1.0)

    __rmul__ = __mul__  # handle float * Hamiltonian

    def serialize_function(self, **kwargs) -> DataFuture:
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
        """Return function parameters"""
        raise NotImplementedError

    def get_app(self) -> Callable:
        raise NotImplementedError


@psiflow.register_serializable
class MixtureHamiltonian(Hamiltonian):
    hamiltonians: list[Hamiltonian]
    coefficients: list[float]

    def __init__(
        self,
        hamiltonians: Sequence[Hamiltonian],
        coefficients: Sequence[float],
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
        apply_apps = [h.get_app() for h in self.hamiltonians]
        reduce_func = partial(
            aggregate_multiple, coefficients=np.array(self.coefficients)
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
        hamiltonian: MixtureHamiltonian
        if len(self.coefficients) != len(hamiltonian.coefficients):
            return False
        for c, h in zip(self.coefficients, self.hamiltonians):
            if (idx := hamiltonian.get_index(h)) is None:
                return False
            if c != hamiltonian.coefficients[idx]:
                return False
        return True

    def __mul__(self, a: float) -> "MixtureHamiltonian":
        return MixtureHamiltonian(
            self.hamiltonians,
            [c * a for c in self.coefficients],
        )

    __rmul__ = __mul__  # handle float * MixtureHamiltonian

    def __len__(self) -> int:
        return len(self.coefficients)

    def __add__(self, hamiltonian: Hamiltonian) -> "MixtureHamiltonian":
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

    def get_indices(self, mixture: "MixtureHamiltonian") -> Optional[tuple[int, ...]]:
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
        self, mixture: "MixtureHamiltonian"
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
        # TODO: iter_named_components?
        names, counts = [], {}
        for h in self.hamiltonians:
            name = h.__class__.__name__
            counts.setdefault(name, 0)
            names.append(f"{name}{counts[name]}")
            counts[name] += 1
        return names

    def serialize(self, **kwargs) -> list[DataFuture]:
        return [h.serialize_function(**kwargs) for h in self.hamiltonians]


@psiflow.register_serializable
@dataclass(frozen=True)
class Zero(Hamiltonian):
    function_name: ClassVar[str] = "ZeroFunction"

    def __init__(self):
        pass

    def get_app(self) -> Callable:
        return partial(apply_threads, function_cls=ZeroFunction)

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if type(hamiltonian) is Zero:
            return True
        return False

    def __mul__(self, a: float) -> "Zero":
        return Zero()

    def __add__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        # (Zero + Hamiltonian) is different from (Hamiltonian + Zero)
        return hamiltonian

    __rmul__ = __mul__  # handle float * Zero


@psiflow.register_serializable
@dataclass(frozen=True)
class EinsteinCrystal(Hamiltonian):
    force_constant: float | AppFuture
    centers: np.ndarray | AppFuture
    volume: float | AppFuture
    function_name: ClassVar[str] = "EinsteinCrystalFunction"

    def get_app(self) -> Callable:
        return partial(
            apply_threads, function_cls=EinsteinCrystalFunction, **self.parameters()
        )

    def parameters(self) -> dict:
        return {
            "force_constant": self.force_constant,
            "centers": self.centers,
            "volume": self.volume,
        }

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if (
            not isinstance(hamiltonian, EinsteinCrystal)
            or not np.allclose(self.force_constant, hamiltonian.force_constant)
            or not np.allclose(self.centers, hamiltonian.centers)
            or not np.isclose(self.volume, hamiltonian.volume)
        ):
            return False
        return True

    @classmethod
    def from_geometry(
        cls, geometry: Geometry | AppFuture, force_constant: float | AppFuture
    ):
        # TODO: this is not immutable?
        centers = get_attribute(geometry, "per_atom", "positions")
        volume = get_attribute(geometry, "volume")
        return cls(force_constant, centers, volume)


@psiflow.register_serializable
@dataclass(frozen=True)
class PlumedHamiltonian(Hamiltonian):
    plumed_input: str | AppFuture
    external: Optional[psiflow._DataFuture] = None
    function_name: ClassVar[str] = "PlumedFunction"

    def __post_init__(self):
        self._prepare_input()
        if isinstance(ext := self.external, (str, Path)):
            ext = File(ext)
        if ext is not None:
            assert ext.filepath in self.plumed_input
        object.__setattr__(self, "external", ext)

    def _prepare_input(self) -> None:
        if isinstance(inp := self.plumed_input, Future):
            app = python_app(remove_comments_printflush, executors=["default_threads"])
            inp = app(inp)
        else:
            inp = remove_comments_printflush(inp)
        object.__setattr__(self, "plumed_input", inp)

    def get_app(self) -> Callable:
        return partial(
            apply_htex,
            function_cls=PlumedFunction,
            inputs=[self.external],  # wait for future
            **self.parameters(),
        )

    def parameters(self) -> dict:
        path = self.external.filepath if self.external is not None else None
        return {"plumed_input": self.plumed_input, "external": path}

    def __eq__(self, other: Hamiltonian) -> bool:
        if (
            not isinstance(other, PlumedHamiltonian)
            or self.plumed_input != other.plumed_input
        ):
            return False
        return True


@psiflow.register_serializable
@dataclass(frozen=True)
class Harmonic(Hamiltonian):
    hessian: np.ndarray | AppFuture
    positions: np.ndarray | AppFuture
    energy: np.ndarray | AppFuture
    function_name: ClassVar[str] = "HarmonicFunction"

    def get_app(self) -> Callable:
        return partial(
            apply_threads, function_cls=HarmonicFunction, **self.parameters()
        )

    def parameters(self) -> dict:
        return {
            "positions": self.positions,
            "energy": self.energy,
            "hessian": self.hessian,
        }

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if (
            not isinstance(hamiltonian, Harmonic)
            or not np.allclose(self.hessian, hamiltonian.hessian)
            or not np.allclose(self.positions, hamiltonian.positions)
            or not np.isclose(self.energy, hamiltonian.energy)
        ):
            return False
        return True

    @classmethod
    def from_geometry(
        cls, geometry: Geometry | AppFuture, hessian: np.ndarray | AppFuture
    ):
        # TODO: this is not immutable?
        positions = get_attribute(geometry, "per_atom", "positions")
        energy = get_attribute(geometry, "energy")
        return cls(hessian, positions, energy)


@psiflow.register_serializable
@dataclass(frozen=True)
class D3Hamiltonian(Hamiltonian):
    method: str | AppFuture
    damping: str | AppFuture = "d3bj"
    function_name: ClassVar[str] = "DispersionFunction"

    def get_app(self) -> Callable:
        # execution-side parameters of function are not included in self.parameters()
        evaluation = psiflow.context().definitions["ModelEvaluation"]
        resources = evaluation.wq_resources(1)
        resources.pop("gpus", None)  # do not request GPU
        return partial(
            apply_modelevaluation,
            function_cls=DispersionFunction,
            parsl_resource_specification=resources,
            **self.parameters(),
            num_threads=resources.get("cores"),
        )

    def parameters(self) -> dict:
        return {"method": self.method, "damping": self.damping}

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if (
            not isinstance(hamiltonian, D3Hamiltonian)
            or self.method != hamiltonian.method
            or self.damping != hamiltonian.damping
        ):
            return False
        return True


@psiflow.register_serializable
@dataclass(frozen=True)
class MACEHamiltonian(Hamiltonian):
    external: psiflow._DataFuture
    kwargs: dict
    function_name: ClassVar[str] = "MACEFunction"

    def __post_init__(self):
        if isinstance(ext := self.external, (str, Path)):
            ext = File(ext)
            object.__setattr__(self, "external", ext)

    def update_kwargs(self, **kwargs: Any) -> None:
        """Specify kwargs for MACECalculator (enable_cueq, head, ..)"""
        self.kwargs |= kwargs

    def get_app(self) -> Callable:
        # execution-side parameters of function are not included in self.parameters()
        evaluation = psiflow.context().definitions["ModelEvaluation"]
        return partial(
            apply_modelevaluation,
            function_cls=MACEFunction,
            parsl_resource_specification=evaluation.wq_resources(1),
            inputs=[self.external],  # wait for future
            **self.parameters(include_env=True),
        )

    def parameters(self, include_env: bool = False) -> dict:
        # TODO: in i-Pi MD, 'ncores', 'dtype', 'device' should be set by the sampling module
        #  (most of them are overwritten in the driver right now)
        evaluation = psiflow.context().definitions["ModelEvaluation"]
        data = {
            "model_path": self.external.filepath,
            "ncores": evaluation.cores_per_task,
            "dtype": "float32",
            "device": "cuda" if evaluation.use_gpu else "cpu",
            "calc_kwargs": self.kwargs,
        }
        if include_env:  # python apps need to set env_vars
            data["env_vars"] = evaluation.env_vars
        return data

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if (
            not isinstance(hamiltonian, MACEHamiltonian)
            or self.external.filepath != hamiltonian.external.filepath
            or self.kwargs != hamiltonian.kwargs
        ):
            return False
        return True


def combine_hamiltonians(hamiltonians: list[Hamiltonian]) -> MixtureHamiltonian:
    return sum(hamiltonians, start=Zero())  # mostly for type hinting
