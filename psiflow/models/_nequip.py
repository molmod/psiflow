from __future__ import annotations  # necessary for type-guarding class methods
from typing import Optional, Union, List
import typeguard
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

try:
    from ase.calculators.calculator import BaseCalculator
except ImportError:  # 3.22.1 and below still use Calculator
    from ase.calculators.calculator import Calculator as BaseCalculator

import parsl
from parsl.executors import WorkQueueExecutor
from parsl.app.app import python_app, bash_app
from parsl.data_provider.files import File

import psiflow
from psiflow.models.base import evaluate_dataset
from psiflow.models import BaseModel
from psiflow.utils import get_active_executor, read_yaml


logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
@dataclass
class NequIPConfig:  # taken from nequip@v0.5.6 full.yaml
    dataset_include_keys: list = field(
        default_factory=lambda: ["total_energy", "forces", "virial"]
    )
    dataset_key_mapping: dict = field(
        default_factory=lambda: {
            "energy": "total_energy",
            "forces": "forces",
            "stress": "virial",
        }
    )
    root: Optional[str] = None
    run_name: str = "training"
    seed: int = 123
    dataset_seed: int = 123
    append: bool = True
    default_dtype: str = "float32"
    allow_tf32: bool = False
    model_builders: Optional[list] = field(
        default_factory=lambda: [
            "SimpleIrrepsConfig",
            "EnergyModel",
            "PerSpeciesRescale",
            "StressForceOutput",
            "RescaleEnergyEtc",
        ]
    )
    r_max: float = 5.0
    num_layers: int = 4
    l_max: int = 1
    parity: bool = True
    num_features: int = 32
    nonlinearity_type: str = "gate"
    resnet: bool = False
    nonlinearity_scalars: Optional[dict] = field(
        default_factory=lambda: {
            "e": "silu",
            "o": "tanh",
        }
    )
    nonlinearity_gates: Optional[dict] = field(
        default_factory=lambda: {
            "e": "silu",
            "o": "tanh",
        }
    )
    num_basis: int = 8
    BesselBasis_trainable: bool = True
    PolynomialCutoff_p: int = 6
    invariant_layers: int = 2
    invariant_neurons: int = 64
    avg_num_neighbors: str = "auto"
    use_sc: bool = True
    dataset: str = "ase"
    dataset_file_name: str = "giggle.xyz"
    dataset_validation: str = "ase"
    dataset_validation_file_name: str = "giggle.xyz"
    chemical_symbols: Optional[list[str]] = field(
        default_factory=lambda: ["X"]
    )  # gets overridden
    wandb: bool = True  # enable by default
    wandb_project: str = "psiflow"
    wandb_group: Optional[str] = None
    wandb_watch: bool = False
    verbose: str = "info"
    log_batch_freq: int = 10
    log_epoch_freq: int = 1
    save_checkpoint_freq: int = -1
    save_ema_checkpoint_freq: int = -1
    n_train: int = 0  # no need to set
    n_val: int = 0  # no need to set
    learning_rate: float = 0.005
    batch_size: int = 5
    validation_batch_size: int = 10
    max_epochs: int = 100000
    train_val_split: str = "random"
    shuffle: bool = True
    metrics_key: str = "validation_loss"
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_use_num_updates: bool = True
    report_init_validation: bool = True
    early_stopping_patiences: Optional[dict] = field(
        default_factory=lambda: {"validation_loss": 100}
    )
    early_stopping_delta: Optional[dict] = field(
        default_factory=lambda: {"validation_loss": 0.002}
    )
    early_stopping_cumulative_delta: bool = False
    early_stopping_lower_bounds: Optional[dict] = field(
        default_factory=lambda: {"LR": 1e-10}
    )
    early_stopping_upper_bounds: Optional[dict] = field(
        default_factory=lambda: {"cumulative_wall": 1e100}
    )
    loss_coeffs: Optional[dict] = field(
        default_factory=lambda: {"forces": 1, "total_energy": [10, "PerAtomMSELoss"]}
    )
    metrics_components: Optional[list] = field(
        default_factory=lambda: [
            ["forces", "mae"],
            ["forces", "rmse"],
            ["forces", "mae", {"PerSpecies": True, "report_per_component": False}],
            ["forces", "rmse", {"PerSpecies": True, "report_per_component": False}],
            ["total_energy", "mae"],
            ["total_energy", "mae", {"PerAtom": True}],
        ]
    )
    optimizer_name: str = "Adam"
    optimizer_amsgrad: bool = False
    optimizer_betas: tuple = tuple([0.9, 0.999])
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0
    max_gradient_norm: Optional[float] = None
    lr_scheduler_name: str = "ReduceLROnPlateau"
    lr_scheduler_patience: int = 200
    lr_scheduler_factor: float = 0.5
    per_species_rescale_scales_trainable: bool = False
    per_species_rescale_shifts_trainable: bool = False
    per_species_rescale_shifts: Optional[str] = "dataset_per_atom_total_energy_mean"
    per_species_rescale_scales: Optional[str] = "dataset_per_species_forces_rms"
    global_rescale_shift: Optional[str] = None
    global_rescale_scale: Optional[str] = None
    global_rescale_shift_trainable: bool = False
    global_rescale_scale_trainable: bool = False


@typeguard.typechecked
@dataclass
class AllegroConfig(NequIPConfig):
    model_builders: Optional[dict] = field(
        default_factory=lambda: [
            "allegro.model.Allegro",
            "PerSpeciesRescale",
            "ForceOutput",
            "RescaleEnergyEtc",
        ]
    )
    parity: str = "o3_full"
    num_layers: int = 1
    env_embed_multiplicity: int = 8
    embed_initial_edge: bool = True
    two_body_latent_mlp_latent_dimensions: Optional[list] = field(
        default_factory=lambda: [32, 64, 128]
    )
    two_body_latent_mlp_nonlinearity: str = "silu"
    two_body_latent_mlp_initialization: str = "uniform"
    mlp_latent_dimensions: Optional[list] = field(default_factory=lambda: [128])
    latent_mlp_nonlinearity: str = "silu"
    latent_mlp_initialization: str = "uniform"
    latent_resnet: bool = True
    env_embed_mlp_latent_dimensions: Optional[list] = field(default_factory=lambda: [])
    env_embed_mlp_nonlinearity: Optional[bool] = None
    env_embed_mlp_initialization: str = "uniform"
    wandb: bool = False
    r_max: float = 5.0


@typeguard.typechecked
def initialize(
    nequip_config: dict,
    stdout: str = "",
    stderr: str = "",
    inputs: List[File] = [],
    outputs: List[File] = [],
) -> str:
    import yaml

    nequip_config["dataset_file_name"] = inputs[0].filepath
    config_str = yaml.dump(dict(nequip_config))
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd = "cd $mytmpdir;"
    command_write = 'echo "{}" > config.yaml;'.format(config_str)
    command_list = [
        command_tmp,
        command_cd,
        command_write,
        "psiflow-train-nequip",
        "--config=config.yaml",
        "--model=None",
        "--init_only;",
        "ls *;",
        "cp undeployed.pth {};".format(outputs[0].filepath),
        "cp config.yaml {};".format(outputs[1].filepath),
    ]
    return " ".join(command_list)


@typeguard.typechecked
def deploy(
    nequip_config: dict,
    inputs: List[File] = [],
    outputs: List[File] = [],
) -> str:
    import yaml

    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd = "cd $mytmpdir;"
    config_str = yaml.dump(dict(nequip_config))
    command_write = 'echo "{}" > config.yaml;'.format(config_str)
    command_list = [
        command_tmp,
        command_cd,
        command_write,
        "psiflow-deploy-nequip",
        "--config=config.yaml",
        "--model={}".format(inputs[0].filepath),
        "--deployed={}".format(outputs[0].filepath),
    ]
    return " ".join(command_list)


def train(
    nequip_config: dict,
    stdout: str = "",
    stderr: str = "",
    inputs: List[File] = [],
    outputs: List[File] = [],
    walltime: float = 1e12,  # infinite by default
    parsl_resource_specification: dict = None,
) -> str:
    import yaml

    nequip_config["dataset_file_name"] = inputs[1].filepath
    nequip_config["validation_dataset"] = "ase"
    nequip_config["validation_dataset_file_name"] = inputs[2].filepath
    config_str = yaml.dump(dict(nequip_config))
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd = "cd $mytmpdir;"
    command_env = "export WANDB_CACHE_DIR=$(pwd);"
    command_write = 'echo "{}" > config.yaml;'.format(config_str)
    command_list = [
        command_tmp,
        command_cd,
        command_env,
        command_write,
        "timeout -s 15 {}s".format(max(walltime - 15, 0)),  # 15 s slack
        "psiflow-train-nequip",
        "--config config.yaml",
        "--model {};".format(inputs[0].filepath),
        "ls;",
        "cp {}/best_model.pth {}".format(
            nequip_config["run_name"], outputs[0].filepath
        ),
    ]
    return " ".join(command_list)


@typeguard.typechecked
class NequIPModel(BaseModel):
    """Container class for NequIP models"""

    def __init__(self, config: Union[dict, NequIPConfig]) -> None:
        if isinstance(config, NequIPConfig):
            config = asdict(config)
        else:
            config = dict(config)
        super().__init__(config)

    def deploy(self) -> None:
        assert self.config_future is not None
        assert self.model_future is not None
        context = psiflow.context()
        self.deploy_future = context.apps(self.__class__, "deploy")(
            self.config_future,
            inputs=[self.model_future],
            outputs=[context.new_file("deployed_", ".pth")],
        ).outputs[0]

    @classmethod
    def create_apps(cls) -> None:
        context = psiflow.context()
        evaluation, training = context[cls]
        training_label = training.name()
        training_walltime = training.max_walltime
        training_ncores = training.cores_per_worker
        if isinstance(get_active_executor(training_label), WorkQueueExecutor):
            training_resource_specification = (
                training.generate_parsl_resource_specification()
            )
        else:
            training_resource_specification = {}
        model_label = evaluation.name()
        model_device = "cuda" if evaluation.gpu else "cpu"
        model_ncores = evaluation.cores_per_worker

        app_initialize = bash_app(initialize, executors=["Default"])
        app_deploy = bash_app(deploy, executors=[training_label])
        context.register_app(cls, "deploy", app_deploy)

        def initialize_wrapped(config_raw, inputs=[]):
            assert len(inputs) == 1
            outputs = [
                context.new_file("model_", ".pth"),
                context.new_file("config_", ".yaml"),
            ]
            init_future = app_initialize(
                config_raw,
                inputs=inputs,
                outputs=outputs,
                stdout=parsl.AUTO_LOGNAME,
                stderr=parsl.AUTO_LOGNAME,
            )
            future = read_yaml(inputs=[init_future.outputs[1]], outputs=[])
            deploy_future = app_deploy(
                future,
                inputs=[init_future.outputs[0]],
                outputs=[context.new_file("deploy_", ".pth")],
            )
            return future, init_future.outputs[0], deploy_future.outputs[0]

        context.register_app(cls, "initialize", initialize_wrapped)

        app_train = bash_app(train, executors=[training_label])

        def train_wrapped(config, inputs=[], outputs=[]):
            outputs = [context.new_file("model_", ".pth")]
            future = app_train(
                config,
                stdout=parsl.AUTO_LOGNAME,
                stderr=parsl.AUTO_LOGNAME,
                inputs=inputs,
                outputs=outputs,
                walltime=training_walltime * 60,
                parsl_resource_specification=training_resource_specification,
            )
            deploy_future = app_deploy(
                config,
                inputs=[future.outputs[0]],
                outputs=[context.new_file("deploy_", ".pth")],
            )
            return future.outputs[0], deploy_future.outputs[0]

        context.register_app(cls, "train", train_wrapped)
        evaluate_unwrapped = python_app(
            evaluate_dataset,
            executors=[model_label],
            cache=False,
        )

        def evaluate_wrapped(inputs=[], outputs=[]):
            return evaluate_unwrapped(
                model_device,
                model_ncores,
                cls.load_calculator,
                inputs=inputs,
                outputs=outputs,
            )

        context.register_app(cls, "evaluate", evaluate_wrapped)

    @classmethod
    def load_calculator(
        cls,
        path_model: Union[Path, str],
        device: str,
    ) -> BaseCalculator:
        from nequip.ase import NequIPCalculator

        return NequIPCalculator.from_deployed_model(
            model_path=path_model,
            device=device,
        )

    @property
    def seed(self) -> int:
        return self.config_raw["seed"]

    @seed.setter
    def seed(self, arg: int) -> None:
        self.config_raw["seed"] = arg
        self.config_raw["dataset_seed"] = arg


@typeguard.typechecked
class AllegroModel(NequIPModel):
    def __init__(self, config: Union[dict, AllegroConfig]) -> None:
        if isinstance(config, AllegroConfig):
            config = asdict(config)
        else:
            config = dict(config)
        super().__init__(config)
