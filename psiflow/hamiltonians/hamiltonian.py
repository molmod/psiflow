from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import Optional

import typeguard
from parsl.app.app import join_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry
from psiflow.hamiltonians.utils import add_contributions

logger = logging.getLogger(__name__)  # logging per module


@join_app
@typeguard.typechecked
def evaluate_batched(
    hamiltonian: Hamiltonian,
    dataset: Dataset,
    length: int,
    batch_size: int,
    outputs: list = [],
):
    from math import ceil

    from psiflow.data import join_frames

    if (batch_size is None) or (batch_size >= length):
        evaluated = [hamiltonian.single_evaluate(dataset)]
    else:
        nbatches = ceil(length / batch_size)
        evaluated = []
        for i in range(nbatches - 1):
            batch = dataset[i * batch_size : (i + 1) * batch_size]
            evaluated.append(hamiltonian.single_evaluate(batch))
        last = dataset[(nbatches - 1) * batch_size :]
        evaluated.append(hamiltonian.single_evaluate(last))

    return join_frames(  # join_app requires returning AppFuture
        inputs=[dataset.extxyz for dataset in evaluated],
        outputs=[outputs[0]],
    )


@typeguard.typechecked
@psiflow.serializable  # otherwise MixtureHamiltonian.hamiltonians is not serialized
class Hamiltonian:
    external: Optional[psiflow._DataFuture]

    def evaluate(self, dataset: Dataset, batch_size: Optional[int] = 100) -> Dataset:
        future = evaluate_batched(
            self,
            dataset,
            dataset.length(),
            batch_size,
            outputs=[
                psiflow.context().new_file("data_", ".xyz")
            ],  # join_app needs outputs kwarg here!
        )
        return Dataset(None, future.outputs[0])

    # mostly for internal use
    def single_evaluate(self, dataset: Dataset) -> Dataset:
        future = self.evaluate_app(
            self.load_calculators,
            inputs=[dataset.extxyz, self.external],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
            **self.parameters,
        )
        return Dataset(None, future.outputs[0])

    def serialize_calculator(self) -> DataFuture:
        raise NotImplementedError

    @property
    def parameters(self: Hamiltonian) -> dict:
        raise NotImplementedError

    @staticmethod
    def load_calculators(
        data: list[Geometry],
        external: Optional[File],
    ) -> tuple:
        raise NotImplementedError

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


@typeguard.typechecked
class Zero(Hamiltonian):
    def single_evaluate(self, dataset: Dataset) -> Dataset:
        return dataset.reset()

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

    def evaluate(self, dataset: Dataset, batch_size: Optional[int] = 100) -> Dataset:
        evaluated = [h.evaluate(dataset) for h in self.hamiltonians]
        future = add_contributions(
            tuple(self.coefficients),
            inputs=[e.extxyz for e in evaluated],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        )
        return Dataset(None, future.outputs[0])

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
