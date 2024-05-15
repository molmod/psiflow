from __future__ import annotations  # necessary for type-guarding class methods

import urllib
from functools import partial
from pathlib import Path
from typing import Any, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.geometry import Geometry
from psiflow.hamiltonians.hamiltonian import Hamiltonian, evaluate_function
from psiflow.utils import dump_json


@typeguard.typechecked
@psiflow.serializable
class MACEHamiltonian(Hamiltonian):
    atomic_energies: dict[str, float]
    external: psiflow._DataFuture

    def __init__(
        self,
        model_future: Union[DataFuture, File],
        atomic_energies: dict[str, float],
    ) -> None:
        super().__init__()
        self.atomic_energies = atomic_energies
        self.external = model_future
        self._create_apps()

    def _create_apps(self):
        evaluation = psiflow.context().definitions["ModelEvaluation"]
        resources = evaluation.wq_resources(1)
        ncores = evaluation.cores_per_worker
        if evaluation.gpu:
            device = "cuda"
        else:
            device = "cpu"
        infused_evaluate = partial(
            evaluate_function,
            ncores=ncores,
            device=device,
            dtype="float32",
        )
        evaluate_app = python_app(infused_evaluate, executors=[evaluation.name])
        if resources is not None:
            self.evaluate_app = partial(
                evaluate_app,
                parsl_resource_specification=resources,
            )
        else:
            self.evaluate_app = evaluate_app

    def serialize_calculator(self) -> DataFuture:
        return dump_json(
            hamiltonian=self.__class__.__name__,
            model_path=str(Path(self.external.filepath).resolve()),
            atomic_energies=self.atomic_energies,
            inputs=[self.external],
            outputs=[psiflow.context().new_file("hamiltonian_", ".json")],
        ).outputs[0]

    @staticmethod
    def deserialize_calculator(
        model_path: str,
        atomic_energies: dict,
        device: str,
        dtype: str,
    ) -> Any:
        import torch

        from psiflow.models.mace_utils import MACECalculator

        calculator = MACECalculator(
            model_path=model_path,
            device=device,
            dtype=dtype,
            atomic_energies=atomic_energies,
        )
        # somehow necessary to override this stuff
        torch_dtype = getattr(torch, dtype)
        calculator.model.to(torch_dtype)
        torch.set_default_dtype(torch_dtype)
        return calculator

    @property
    def parameters(self: Hamiltonian) -> dict:
        return {"atomic_energies": self.atomic_energies}

    def __eq__(self, hamiltonian) -> bool:
        if type(hamiltonian) is not MACEHamiltonian:
            return False
        if self.external.filepath != hamiltonian.external.filepath:
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
        data: list[Geometry],
        model_future: Union[DataFuture, File],
        atomic_energies: dict,
        ncores: int,
        device: str,
        dtype: str,  # float64 for optimizations
    ) -> tuple[list[Any], np.ndarray]:
        import numpy as np
        import torch

        from psiflow.models.mace_utils import MACECalculator

        calculator = MACECalculator(
            model_future.filepath,
            device=device,
            dtype=dtype,
            atomic_energies=atomic_energies,
        )
        # somehow necessary to set this again
        torch_dtype = getattr(torch, dtype)
        calculator.model.to(torch_dtype)
        torch.set_default_dtype(torch_dtype)
        index_mapping = np.zeros(len(data), dtype=int)
        torch.set_num_threads(ncores)
        return [calculator], index_mapping


def get_mace_mp0(size: str = "small") -> MACEHamiltonian:
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
    return MACEHamiltonian(parsl_file, {})


def get_mace_cc() -> MACEHamiltonian:
    url = "https://github.com/ACEsuit/mace/raw/main/mace/calculators/foundations_models/ani500k_large_CC.model"
    parsl_file = psiflow.context().new_file("mace_mp_", ".pth")
    urllib.request.urlretrieve(
        url,
        parsl_file.filepath,
    )
    return MACEHamiltonian(parsl_file, {})
