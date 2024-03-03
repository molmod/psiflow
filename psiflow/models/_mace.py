from __future__ import annotations  # necessary for type-guarding class methods

import logging
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import numpy as np
import typeguard
from ase.calculators.calculator import Calculator
from parsl.app.app import bash_app, python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.data import FlowAtoms
from psiflow.hamiltonians.hamiltonian import Hamiltonian, evaluate_function
from psiflow.models.model import Model

logger = logging.getLogger(__name__)  # logging per module


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


@typeguard.typechecked
@dataclass
class MACEConfig:
    name: str = "mace"
    seed: int = 0
    log_dir: str = ""  # gets overwritten
    model_dir: str = ""
    checkpoints_dir: str = ""
    results_dir: str = ""
    downloads_dir: str = ""
    device: str = "cuda"
    default_dtype: str = "float32"
    log_level: str = "INFO"
    error_table: str = "PerAtomRMSE"
    model: str = "MACE"
    r_max: float = 5.0
    radial_type: str = "bessel"
    num_radial_basis: int = 8
    num_cutoff_basis: int = 5
    interaction: str = "RealAgnosticResidualInteractionBlock"
    interaction_first: str = "RealAgnosticResidualInteractionBlock"
    max_ell: int = 3
    correlation: int = 3
    num_interactions: int = 2
    MLP_irreps: str = "16x0e"
    radial_MLP: str = "[64, 64, 64]"
    num_channels: int = 16  # hidden_irreps is determined by num_channels and max_L
    max_L: int = 1
    gate: str = "silu"
    scaling: str = "rms_forces_scaling"
    avg_num_neighbors: Optional[float] = None
    compute_avg_num_neighbors: bool = True
    compute_stress: bool = True
    compute_forces: bool = True
    train_file: Optional[str] = None
    valid_file: Optional[str] = None
    # model_dtype: str = "float32"
    valid_fraction: float = 1e-12  # never split training set
    test_file: Optional[str] = None
    E0s: Optional[str] = "average"
    energy_key: str = "energy"
    forces_key: str = "forces"
    virials_key: str = "virials"
    stress_key: str = "stress"
    dipole_key: str = "dipole"
    charges_key: str = "charges"
    loss: str = "weighted"
    forces_weight: float = 1
    swa_forces_weight: float = 1
    energy_weight: float = 10
    swa_energy_weight: float = 100
    virials_weight: float = 0
    swa_virials_weight: float = 0
    stress_weight: float = 0
    swa_stress_weight: float = 0
    dipole_weight: float = 0
    swa_dipole_weight: float = 0
    config_type_weights: str = '{"Default":1.0}'
    huber_delta: float = 0.01
    optimizer: str = "adam"
    batch_size: int = 1
    valid_batch_size: int = 8
    lr: float = 0.01
    swa_lr: float = 0.001
    weight_decay: float = 5e-7
    amsgrad: bool = True
    scheduler: str = "ReduceLROnPlateau"
    lr_factor: float = 0.8
    scheduler_patience: int = 50
    lr_scheduler_gamma: float = 0.9993
    swa: bool = False
    start_swa: int = int(1e12)  # never start swa
    ema: bool = False
    ema_decay: float = 0.99
    max_num_epochs: int = int(1e6)
    patience: int = 2048
    eval_interval: int = 2
    keep_checkpoints: bool = False
    restart_latest: bool = False
    save_cpu: bool = True
    clip_grad: Optional[float] = 10
    wandb: bool = False
    wandb_project: Optional[str] = "psiflow"
    wandb_group: Optional[str] = None
    wandb_name: str = "mace_training"

    @staticmethod
    def serialize(config: dict):
        store_true_keys = [
            "amsgrad",
            "swa",
            "ema",
            "keep_checkpoints",
            "restart_latest",
            "save_cpu",
            "wandb",
        ]
        config_str = ""
        for key, value in config.items():
            if type(value) is bool:
                if key in store_true_keys:
                    if value:
                        config_str += "--{} ".format(key)
                    continue
            if value is None:
                continue

            # get rid of any whitespace in str(value), as e.g. with lists
            config_str += "".join("--{}={}".format(key, value).split()) + " "
        return config_str


# @typeguard.typechecked
def initialize(
    mace_config: dict,
    command_train: str,
    stdout: str = "",
    stderr: str = "",
    inputs: list[File] = [],
    outputs: list[File] = [],
) -> str:
    from psiflow.models._mace import MACEConfig

    assert len(inputs) == 1
    mace_config["train_file"] = inputs[0].filepath
    mace_config["valid_file"] = None
    config_str = MACEConfig.serialize(mace_config)
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd = "cd $mytmpdir;"
    command_list = [
        command_tmp,
        command_cd,
        command_train,
        config_str,
        ";",
        "cp model.pth {};".format(outputs[0].filepath),
    ]
    return " ".join(command_list)


def train(
    mace_config: dict,
    command_train: str,
    stdout: str = "",
    stderr: str = "",
    inputs: list[File] = [],
    outputs: list[File] = [],
) -> str:
    from psiflow.models._mace import MACEConfig

    assert len(inputs) == 3
    mace_config["train_file"] = inputs[1].filepath
    mace_config["valid_file"] = inputs[2].filepath
    mace_config["device"] = "cuda"
    config_str = MACEConfig.serialize(mace_config)
    config_str += "--initialized_model={} ".format(inputs[0].filepath)

    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd = "cd $mytmpdir;"
    command_list = [
        command_tmp,
        command_cd,
        command_train,
        config_str,
        ";",
        "cp model.pth {};".format(outputs[0].filepath),
    ]
    return " ".join(command_list)


@typeguard.typechecked
class MACEModel(Model):
    def __init__(self, **config) -> None:
        super().__init__()
        config = MACEConfig(**config)
        assert not config.swa, "usage of SWA is currently not supported"
        assert config.save_cpu  # assert model is saved to CPU after training
        config.device = "cpu"
        self.config = config

        # initialize apps
        evaluation = psiflow.context()["ModelEvaluation"].name
        training = psiflow.context()["ModelTraining"].name
        command_init = psiflow.context()["ModelTraining"].train_command(True)
        command_train = psiflow.context()["ModelTraining"].train_command(False)

        app_initialize = bash_app(initialize, executors=[evaluation])
        app_train = bash_app(train, executors=[training])
        self._initialize = partial(app_initialize, command_train=command_init)
        self._train = partial(app_train, command_train=command_train)

    def create_hamiltonian(self):
        return MACEHamiltonian(self.model_future, self.atomic_energies)

    @property
    def seed(self) -> int:
        return self.config.seed

    @seed.setter
    def seed(self, arg: int) -> None:
        self.config.seed = arg
