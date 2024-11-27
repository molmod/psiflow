from __future__ import annotations  # necessary for type-guarding class methods

import logging
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional, Union

import typeguard
from parsl.app.app import bash_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.hamiltonians import MACEHamiltonian
from psiflow.models.model import Model

logger = logging.getLogger(__name__)  # logging per module


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
    default_dtype: str = "float32"  # default: float64
    # distributed: bool = False  # this is automatically set based on execution config
    log_level: str = "INFO"
    error_table: str = "PerAtomRMSE"
    model: str = "MACE"
    r_max: float = 5.0
    radial_type: str = "bessel"
    num_radial_basis: int = 8
    num_cutoff_basis: int = 5
    pair_repulsion: bool = False
    distance_transform: Optional[str] = None
    interaction: str = "RealAgnosticResidualInteractionBlock"
    interaction_first: str = "RealAgnosticResidualInteractionBlock"
    max_ell: int = 3
    correlation: int = 3
    num_interactions: int = 2
    MLP_irreps: str = "16x0e"
    radial_MLP: str = "[64, 64, 64]"
    num_channels: int = 16  # default: 128 channels
    max_L: int = 1
    gate: str = "silu"
    scaling: str = "rms_forces_scaling"
    avg_num_neighbors: Optional[float] = None
    compute_avg_num_neighbors: bool = True
    compute_stress: bool = True  # default: False
    compute_forces: bool = True
    train_file: Optional[str] = None
    valid_file: Optional[str] = None
    valid_fraction: float = 1e-12  # never split training set
    test_file: Optional[str] = None
    num_workers: int = 0
    pin_memory: bool = True
    E0s: Optional[str] = "average"
    energy_key: str = "energy"
    forces_key: str = "forces"
    virials_key: str = "virials"
    stress_key: str = "stress"
    dipole_key: str = "dipole"
    charges_key: str = "charges"
    loss: str = "weighted"
    forces_weight: float = 1  # default: 100
    swa_forces_weight: float = 1
    energy_weight: float = 10  # default: 1
    swa_energy_weight: float = 100
    virials_weight: float = 0  # default: 1
    swa_virials_weight: float = 0  # default: 10
    stress_weight: float = 0  # default: 1
    swa_stress_weight: float = 0  # default: 10
    config_type_weights: str = '{"Default":1.0}'
    huber_delta: float = 0.01
    optimizer: str = "adam"
    batch_size: int = 10
    valid_batch_size: int = 10
    lr: float = 0.01
    swa_lr: float = 0.001
    weight_decay: float = 5e-7
    amsgrad: bool = True
    scheduler: str = "ReduceLROnPlateau"
    lr_factor: float = 0.8
    scheduler_patience: int = 50
    lr_scheduler_gamma: float = 0.9993
    swa: bool = False  # default: False
    start_swa: Optional[int] = int(1e10)  # never start swa
    ema: bool = False
    ema_decay: float = 0.99
    max_num_epochs: int = 2048
    patience: int = 2048
    eval_interval: int = 2
    keep_checkpoints: bool = False
    restart_latest: bool = False
    save_cpu: bool = True  # default: False
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
            "pair_repulsion",
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
    parsl_resource_specification: Optional[dict] = None,
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
    env_vars: dict = {},
    inputs: list[File] = [],
    outputs: list[File] = [],
    parsl_resource_specification: Optional[dict] = None,
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
    command_env = ""
    for key, value in env_vars.items():
        command_env += " export {}={}; ".format(key, value)
    command_list = [
        command_tmp,
        command_cd,
        command_env,
        command_train,
        config_str,
        ";",
        "cp model.pth {};".format(outputs[0].filepath),
    ]
    return " ".join(command_list)


@typeguard.typechecked
@psiflow.serializable
class MACE(Model):
    _config: dict
    model_future: Optional[psiflow._DataFuture]
    atomic_energies: dict[str, Union[float, AppFuture]]

    def __init__(self, **config) -> None:
        config = MACEConfig(**config)  # validate input
        config.save_cpu = True  # assert model is saved to CPU after training
        config.device = "cpu"
        if not config.swa:
            config.start_swa = config.max_num_epochs  # otherwise he fails to read
        self._config = asdict(config)

        self.model_future = None
        self.atomic_energies = {}

        self._create_apps()

    def _create_apps(self):  # initialize apps
        evaluation = psiflow.context().definitions["ModelEvaluation"]
        training = psiflow.context().definitions["ModelTraining"]
        command_init = training.train_command(True)
        command_train = training.train_command(False)

        app_initialize = bash_app(initialize, executors=[evaluation.name])
        resources_init = evaluation.wq_resources(1)
        if resources_init is not None:
            resources_init["running_time_min"] = 30  # at least 30 mins for init?
        app_train = bash_app(train, executors=[training.name])
        resources_train = training.wq_resources()
        self._initialize = partial(
            app_initialize,
            command_train=command_init,
            parsl_resource_specification=resources_init,
        )
        self._train = partial(
            app_train,
            command_train=command_train,
            env_vars=training.env_vars,
            parsl_resource_specification=resources_train,
        )

    @property
    def seed(self) -> int:
        return self._config["seed"]

    @seed.setter
    def seed(self, arg: int) -> None:
        self._config["seed"] = arg

    def create_hamiltonian(self) -> MACEHamiltonian:
        assert self.model_future is not None

        # wait for atomic energy calculations if necessary:
        for element in list(self.atomic_energies):
            value = self.atomic_energies[element]
            if isinstance(value, AppFuture):
                self.atomic_energies[element] = value.result()
        return MACEHamiltonian(self.model_future, self.atomic_energies)
