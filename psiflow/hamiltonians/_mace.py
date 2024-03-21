from __future__ import annotations  # necessary for type-guarding class methods

from functools import partial
from typing import Union

import numpy as np
from ase.calculators.calculator import Calculator
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.data import FlowAtoms
from psiflow.hamiltonians.hamiltonian import Hamiltonian, evaluate_function
from psiflow.models import MACEModel


class MACEHamiltonian(Hamiltonian):
    def __init__(
        self,
        model_future: Union[DataFuture, File],
        atomic_energies: dict[str, float],
    ) -> None:
        super().__init__()
        self.atomic_energies = atomic_energies
        self.input_files = [model_future]

        executor_name = psiflow.context()["ModelEvaluation"].name
        ncores = psiflow.context()["ModelEvaluation"].cores_per_worker
        if psiflow.context()["ModelEvaluation"].gpu:
            device = "cuda"
        else:
            device = "cpu"
        infused_evaluate = partial(
            evaluate_function,
            ncores=ncores,
            device=device,
            dtype="float32",
        )
        self.evaluate_app = python_app(infused_evaluate, executors=[executor_name])

    @property
    def parameters(self: Hamiltonian) -> dict:
        return {"atomic_energies": self.atomic_energies}

    def __eq__(self, hamiltonian) -> bool:
        if type(hamiltonian) is not MACEHamiltonian:
            return False
        if self.input_files[0] != hamiltonian.input_files[0]:
            return False
        if len(self.atomic_energies) != len(hamiltonian.atomic_energies):
            return False
        for symbol, energy in self.atomic_energies:
            if not np.allclose(
                energy,
                hamiltonian.atomic_energies[symbol],
            ):
                return False
        return True

    @staticmethod
    def load_calculators(
        data: list[FlowAtoms],
        model_future: Union[DataFuture, File],
        atomic_energies: dict,
        ncores: int,
        device: str,
        dtype: str,  # float64 for optimizations
    ) -> tuple[list[Calculator, np.ndarray]]:
        import numpy as np
        import torch

        from psiflow.models.mace_utils import MACECalculator

        calculator = MACECalculator(
            model_future,
            device=device,
            dtype=dtype,
            atomic_energies=atomic_energies,
        )
        index_mapping = np.zeros(len(data), dtype=int)
        torch.set_num_threads(ncores)
        return [calculator], index_mapping

    @classmethod
    def from_model(cls, model: MACEModel):
        return cls(model.model_future, model.atomic_energies)
