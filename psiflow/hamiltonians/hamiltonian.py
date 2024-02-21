from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import Callable, Optional, Union

import numpy as np
import typeguard
from ase import Atoms
from parsl.app.app import join_app, python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, FlowAtoms
from psiflow.utils import copy_app_future

from .utils import EinsteinCalculator

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


@typeguard.typechecked
class EinsteinCrystal(Hamiltonian):
    """Dummy hamiltonian which represents a simple harmonic interaction"""

    def __init__(self, atoms: Union[FlowAtoms, AppFuture], force_constant: float):
        super().__init__()
        self.reference_geometry = copy_app_future(atoms)
        self.force_constant = force_constant
        self.input_files = []

        self.evaluate_app = python_app(evaluate_function, executors=["default_threads"])

    @property
    def parameters(self: Hamiltonian) -> dict:
        return {
            "reference_geometry": self.reference_geometry,
            "force_constant": self.force_constant,
        }

    @staticmethod
    def load_calculators(
        data: list[FlowAtoms],
        reference_geometry: Atoms,
        force_constant: float,
    ) -> tuple[list[EinsteinCalculator], np.ndarray]:
        import numpy as np

        from psiflow.hamiltonians.utils import EinsteinCalculator

        assert sum([len(a) == len(reference_geometry) for a in data])
        assert sum([np.all(a.numbers == reference_geometry.numbers) for a in data])
        calculators = [
            EinsteinCalculator(reference_geometry.get_positions(), force_constant)
        ]
        index_mapping = np.zeros(len(data), dtype=int)
        return calculators, index_mapping
