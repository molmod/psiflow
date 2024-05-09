from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import Callable, Optional

import typeguard
from parsl.app.app import join_app, python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def evaluate_function(
    load_calculators: Callable,
    inputs: list = [],
    outputs: list = [],
    parsl_resource_specification: dict = {},
    **parameters,  # dict values can be futures, so app must wait for those
) -> None:
    import numpy as np
    from ase import Atoms

    from psiflow.data import _read_frames, _write_frames
    from psiflow.geometry import NullState

    assert len(inputs) >= 1
    assert len(outputs) == 1
    states = _read_frames(inputs=[inputs[0]])
    calculators, index_mapping = load_calculators(states, inputs[1], **parameters)
    for i, state in enumerate(states):
        if state == NullState:
            continue
        calculator = calculators[index_mapping[i]]
        calculator.reset()
        atoms = Atoms(
            numbers=state.per_atom.numbers,
            positions=state.per_atom.positions,
            cell=state.cell,
            pbc=state.periodic,
        )
        atoms.calc = calculator
        state.energy = atoms.get_potential_energy()
        state.per_atom.forces[:] = atoms.get_forces()
        if state.periodic:
            try:  # some models do not have stress support
                stress = atoms.get_stress(voigt=False)
            except Exception as e:
                print(e)
                stress = np.zeros((3, 3))
            state.stress = stress
    _write_frames(*states, outputs=[outputs[0]])


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


@typeguard.typechecked
def _add_contributions(
    coefficients: tuple[float, ...],
    inputs: list = [],
    outputs: list = [],
) -> None:
    import copy

    from psiflow.data import _read_frames, _write_frames

    contributions = [_read_frames(inputs=[i]) for i in inputs]
    assert len(contributions) == len(coefficients)
    length = len(contributions[0])
    for contribution in contributions:
        assert len(contribution) == length

    data = []
    for i in range(length):
        geometries = [contribution[i] for contribution in contributions]
        energy_list = [geometry.energy for geometry in geometries]
        forces_list = [geometry.per_atom.forces for geometry in geometries]

        energy = sum([energy_list[i] * c for i, c in enumerate(coefficients)])
        forces = sum([forces_list[i] * c for i, c in enumerate(coefficients)])

        geometry = copy.deepcopy(geometries[0])
        geometry.energy = energy
        geometry.per_atom.forces[:] = forces

        if geometry.periodic:
            stress_list = [g.stress for g in geometries]
            stress = sum([stress_list[i] * c for i, c in enumerate(coefficients)])
            geometry.stress = stress
        data.append(geometry)
    _write_frames(*data, outputs=[outputs[0]])


add_contributions = python_app(_add_contributions, executors=["default_threads"])
