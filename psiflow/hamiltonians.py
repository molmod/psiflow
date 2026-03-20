import urllib
from functools import partial
from pathlib import Path
from typing import Optional, Union, Callable, Sequence

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
from psiflow.utils.apps import copy_app_future, get_attribute
from psiflow.utils.io import dump_json


# TODO: remove excess future making


apply_threads = python_app(_apply, executors=["default_threads"])
apply_htex = python_app(_apply, executors=["default_htex"])
apply_modelevaluation = python_app(_apply, executors=["ModelEvaluation"])


# TODO: why have the Computable class?
class Hamiltonian(Computable):
    outputs: tuple = ("energy", "forces", "stress")
    batch_size = 1000
    app: Callable
    function_name: str

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
class Zero(Hamiltonian):
    function_name: str = "ZeroFunction"

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
class EinsteinCrystal(Hamiltonian):
    # TODO: logic not consistent depending on Geometry | AppFuture
    reference_geometry: Geometry | AppFuture
    force_constant: float
    function_name: str = "EinsteinCrystalFunction"

    def __init__(self, geometry: Union[Geometry, AppFuture], force_constant: float):
        super().__init__()
        self.reference_geometry = copy_app_future(geometry)
        self.force_constant = force_constant
        self.external = None  # needed

    def get_app(self) -> Callable:
        return partial(
            apply_threads, function_cls=EinsteinCrystalFunction, **self.parameters()
        )

    def parameters(self) -> dict:
        return {
            "force_constant": self.force_constant,
            "centers": get_attribute(self.reference_geometry, "per_atom", "positions"),
            "volume": get_attribute(self.reference_geometry, "volume"),
        }

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if (
            not isinstance(hamiltonian, EinsteinCrystal)
            or not np.allclose(self.force_constant, hamiltonian.force_constant)
            or self.reference_geometry != hamiltonian.reference_geometry
        ):
            return False
        return True


@psiflow.register_serializable
class PlumedHamiltonian(Hamiltonian):
    plumed_input: str  # TODO: or future?
    external: Optional[psiflow._DataFuture]
    function_name: str = "PlumedFunction"

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

    def get_app(self) -> Callable:
        return partial(apply_htex, function_cls=PlumedFunction, **self.parameters())

    def parameters(self) -> dict:
        if self.external is not None:  # ensure parameters depends on self.external
            external = copy_app_future(self.external.filepath, inputs=[self.external])
        else:
            external = None
        return {"plumed_input": self.plumed_input, "external": external}

    def __eq__(self, other: Hamiltonian) -> bool:
        if (
            not isinstance(other, PlumedHamiltonian)
            or self.plumed_input != other.plumed_input
        ):
            return False
        return True


@psiflow.register_serializable
class Harmonic(Hamiltonian):
    reference_geometry: Geometry | AppFuture
    hessian: np.ndarray | AppFuture
    function_name: str = "HarmonicFunction"

    def __init__(
        self,
        reference_geometry: Geometry | AppFuture,
        hessian: np.ndarray | AppFuture,
    ):
        # TODO: why not copy_app_future(geometry) like others?
        self.reference_geometry = reference_geometry
        self.hessian = hessian

    def get_app(self) -> Callable:
        return partial(
            apply_threads, function_cls=HarmonicFunction, **self.parameters()
        )

    def parameters(self) -> dict:
        positions = get_attribute(self.reference_geometry, "per_atom", "positions")
        energy = get_attribute(self.reference_geometry, "energy")
        return {
            "positions": positions,
            "energy": energy,
            "hessian": self.hessian,
        }

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if type(hamiltonian) is not Harmonic:
            return False
        hamiltonian: Harmonic
        if hamiltonian.reference_geometry != self.reference_geometry:
            return False

        # TODO: why this check? Is it not always an ndarray?
        # slightly different check for numpy arrays
        is_array0 = type(hamiltonian.hessian) is np.ndarray
        is_array1 = type(self.hessian) is np.ndarray
        if is_array0 and is_array1:
            equal = np.allclose(hamiltonian.hessian, self.hessian)
        else:
            equal = hamiltonian.hessian == self.hessian
        if not equal:
            return False
        return True


@psiflow.register_serializable
class D3Hamiltonian(Hamiltonian):
    method: str
    damping: str
    function_name: str = "DispersionFunction"

    def __init__(self, method: str, damping: str = "d3bj"):
        self.method, self.damping = method, damping

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
class MACEHamiltonian(Hamiltonian):
    external: psiflow._DataFuture
    atomic_energies: dict[str, float | Future]
    function_name: str = "MACEFunction"

    def __init__(
        self,
        external: Union[Path, str, psiflow._DataFuture],
        atomic_energies: dict[str, float | Future],
    ):
        self.atomic_energies = atomic_energies
        if isinstance(external, (str, Path)):
            self.external = File(external)
        else:
            self.external = external

    def get_app(self) -> Callable:
        # TODO: this is a python app -> env_vars/cores/.. needed
        # execution-side parameters of function are not included in self.parameters()
        evaluation = psiflow.context().definitions["ModelEvaluation"]
        resources = evaluation.wq_resources(1)
        return partial(
            apply_modelevaluation,
            function_cls=MACEFunction,
            parsl_resource_specification=resources,
            **self.parameters(),
        )

    def parameters(self) -> dict:
        # TODO: Why is the future copy needed? Can we not pass the File/DataFuture directly?
        #  no because the MACEFunction expects a str to point at the model, not a File
        #  this way, the filepath only becomes available after self.external is resolved
        #  so _apply does not start early (DataFuture.filepath is not a future itself)
        #  .
        #  however, we can avoid the copy by piping self.external into a new 'inputs' argument
        #  for _apply, so it waits correctly for all futures
        model_path = copy_app_future(self.external.filepath, inputs=[self.external])
        evaluation = psiflow.context().definitions["ModelEvaluation"]

        print(self.external)
        print(self.external.filepath)
        print(model_path)

        return {
            "model_path": model_path,
            "atomic_energies": self.atomic_energies,
            "ncores": evaluation.cores_per_task,
            "dtype": "float32",
            "device": "gpu" if evaluation.use_gpu else "cpu",
            "env_vars": evaluation.env_vars,
        }

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        if type(hamiltonian) is not MACEHamiltonian:
            return False
        hamiltonian: MACEHamiltonian
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
    def mace_mp0(cls, size: str = "small") -> "MACEHamiltonian":
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
    def mace_cc(cls) -> "MACEHamiltonian":
        url = "https://github.com/molmod/psiflow/raw/main/examples/data/ani500k_cc_cpu.model"
        parsl_file = psiflow.context().new_file("mace_mp_", ".pth")
        urllib.request.urlretrieve(
            url,
            parsl_file.filepath,
        )
        return cls(parsl_file, {})


def combine_hamiltonians(hamiltonians: list[Hamiltonian]) -> MixtureHamiltonian:
    return sum(hamiltonians, start=Zero())  # mostly for type hinting
