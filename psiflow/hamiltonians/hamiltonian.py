from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import Callable, Optional

import typeguard
from parsl.app.app import join_app, python_app

import psiflow
from psiflow.data import Dataset, FlowAtoms

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

    from psiflow.data import app_join_dataset

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

    return app_join_dataset(  # join_app requires returning AppFuture
        inputs=[dataset.data_future for dataset in evaluated],
        outputs=[outputs[0]],
    )


@typeguard.typechecked
def evaluate_function(
    load_calculators: Callable,
    inputs: list = [],
    outputs: list = [],
    **parameters,  # dict values can be futures, so app must wait for those
) -> None:
    import numpy as np

    from psiflow.data import read_dataset, write_dataset

    assert len(inputs) >= 1
    assert len(outputs) == 1
    data = read_dataset(slice(None), inputs=[inputs[0]])
    calculators, index_mapping = load_calculators(data, *inputs[1:], **parameters)
    for i, atoms in enumerate(data):
        calculator = calculators[index_mapping[i]]
        calculator.reset()
        atoms.reset()
        atoms.calc = calculator
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        if atoms.pbc.any():
            try:  # some models do not have stress support
                stress = atoms.get_stress(voigt=False)
            except Exception as e:
                print(e)
                stress = np.zeros((3, 3))
            atoms.info["stress"] = stress
        else:  # remove if present
            atoms.info.pop("stress", None)
        atoms.calc = None
    write_dataset(data, outputs=[outputs[0]])


@typeguard.typechecked
class Hamiltonian:
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
        return Dataset(None, data_future=future.outputs[0])

    # mostly for internal use
    def single_evaluate(self, dataset: Dataset) -> Dataset:
        future = self.evaluate_app(
            self.load_calculators,
            inputs=[dataset.data_future, *self.input_files],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
            **self.parameters,
        )
        return Dataset(None, data_future=future.outputs[0])

    @property
    def parameters(self: Hamiltonian) -> dict:
        raise NotImplementedError

    @staticmethod
    def load_calculators(data: list[FlowAtoms], *inputs, **parameters) -> tuple:
        raise NotImplementedError

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        raise NotImplementedError

    def __mul__(self, a: float) -> Hamiltonian:
        return MixtureHamiltonian([self], [a])

    def __add__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        if type(hamiltonian) is MixtureHamiltonian:
            return hamiltonian.__add__(self)
        return 1.0 * self + 1.0 * hamiltonian

    __rmul__ = __mul__  # handle float * Hamiltonian


@typeguard.typechecked
def add_contributions(
    coefficients: tuple[float, ...],
    inputs: list = [],
    outputs: list = [],
) -> None:
    from psiflow.data import read_dataset, write_dataset

    contributions = [read_dataset(slice(None), inputs=[i]) for i in inputs]
    assert len(contributions) == len(coefficients)
    length = len(contributions[0])
    for contribution in contributions:
        assert len(contribution) == length

    data = []
    for i in range(length):
        atoms_list = [contribution[i] for contribution in contributions]
        energy_list = [atoms.info["energy"] for atoms in atoms_list]
        forces_list = [atoms.arrays["forces"] for atoms in atoms_list]

        energy = sum([energy_list[i] * c for i, c in enumerate(coefficients)])
        forces = sum([forces_list[i] * c for i, c in enumerate(coefficients)])
        atoms = atoms_list[0].copy()
        atoms.info["energy"] = energy
        atoms.arrays["forces"] = forces

        if atoms_list[0].pbc.any():
            stress_list = [atoms.info["stress"] for atoms in atoms_list]
            stress = sum([stress_list[i] * c for i, c in enumerate(coefficients)])
            atoms.info["stress"] = stress
        data.append(atoms)
    write_dataset(data, outputs=[outputs[0]])


app_add_contributions = python_app(add_contributions, executors=["default_threads"])


class MixtureHamiltonian(Hamiltonian):
    def __init__(
        self,
        hamiltonians: list[Hamiltonian, ...],
        coefficients: list[float, ...],
    ) -> None:
        self.hamiltonians = hamiltonians
        self.coefficients = coefficients

    def evaluate(self, dataset: Dataset, batch_size: Optional[int] = 100) -> Dataset:
        evaluated = [h.evaluate(dataset) for h in self.hamiltonians]
        future = app_add_contributions(
            tuple(self.coefficients),
            inputs=[e.data_future for e in evaluated],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        )
        return Dataset(None, data_future=future.outputs[0])

    def __eq__(self, hamiltonian: Hamiltonian) -> bool:
        raise NotImplementedError

    def __add__(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        if type(hamiltonian) is not MixtureHamiltonian:
            try:
                index = self.hamiltonians.index(hamiltonian)
                self.coefficients[index] += 1.0
            except ValueError:
                self.hamiltonians.append(hamiltonian)
                self.coefficients.append(1.0)
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
