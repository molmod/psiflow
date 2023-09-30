from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Any, Dict
import typeguard
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

try:
    from ase.calculators.calculator import BaseCalculator
except ImportError: # 3.22.1 and below still use Calculator
    from ase.calculators.calculator import Calculator as BaseCalculator

import parsl
from parsl.executors import WorkQueueExecutor
from parsl.app.app import python_app, bash_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.models import BaseModel
from psiflow.models.base import evaluate_dataset
from psiflow.data import Dataset
from psiflow.utils import get_active_executor, read_yaml, \
        copy_data_future


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
@dataclass
class MACEConfig:
    train_file: Optional[str] = None
    valid_file: Optional[str] = None
    name: str = 'mace'
    seed: int = 0
    log_level: str = 'INFO'
    log_dir: str = '' # gets overwritten
    model_dir: str = ''
    results_dir: str = ''
    downloads_dir: str = ''
    checkpoints_dir: str = ''
    error_table: str = 'PerAtomRMSE'
    model: str = 'MACE'
    r_max: float = 5.0
    num_radial_basis: int = 8
    num_cutoff_basis: int = 5
    radial_MLP: str = '[64, 64, 64]'
    interaction: str = 'RealAgnosticResidualInteractionBlock'
    interaction_first: str = 'RealAgnosticResidualInteractionBlock'
    max_ell: int = 3
    correlation: int = 3
    num_interactions: int = 2
    MLP_irreps: str = '16x0e'
    num_channels: int = 16 # hidden_irreps is determined by num_channels and max_L
    max_L: int = 1
    gate: str = 'silu'
    scaling: str = 'rms_forces_scaling'
    avg_num_neighbors: Optional[float] = None
    compute_avg_num_neighbors: bool = True
    compute_stress: bool = True
    compute_forces: bool = True
    device: str = 'cuda'
    default_dtype: str = 'float32'
    model_dtype: str = 'float32'
    config_type_weights: str = '{"Default":1.0}'
    valid_fraction: float = 1e-12 # never split training set
    test_file: Optional[str] = None
    E0s: Optional[str] = 'average'
    energy_key: str = 'energy'
    forces_key: str = 'forces'
    stress_key: str = 'stress'
    virials_key: str = 'virials'
    dipole_key: str = 'dipole'
    charges_key: str = 'charges'
    loss: str = 'weighted'
    forces_weight: float = 1
    swa_forces_weight: float = 1
    energy_weight: float = 10
    swa_energy_weight: float = 100
    stress_weight: float = 0
    swa_stress_weight: float = 0
    optimizer: str = 'adam'
    batch_size: int = 1
    valid_batch_size: int = 8
    lr: float = 0.01
    swa_lr: float = 0.001
    weight_decay: float = 5e-7
    amsgrad: bool = True
    scheduler: str = 'ReduceLROnPlateau'
    lr_factor: float = 0.8
    scheduler_patience: int = 50
    lr_scheduler_gamma: float = 0.9993
    swa: bool = False
    start_swa: int = int(1e12) # never start swa
    ema: bool = False
    ema_decay: float = 0.99
    max_num_epochs: int = int(1e6)
    patience: int = 2048
    eval_interval: int = 2
    keep_checkpoints: bool = False
    restart_latest: bool = False
    save_cpu: bool = True
    clip_grad: Optional[float] = 10
    wandb: bool = True
    wandb_project: str = 'psiflow'
    wandb_group: Optional[str] = None
    wandb_name: str = 'mace_training'
    wandb_log_hypers: list = field(default_factory=lambda: [
            "num_channels",
            "max_L",
            "correlation",
            "lr",
            "swa_lr",
            "weight_decay",
            "batch_size",
            "max_num_epochs",
            "start_swa",
            "energy_weight",
            "forces_weight",
            ])


@typeguard.typechecked
def initialize(
        mace_config: dict,
        stdout: str = '',
        stderr: str = '',
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> str:
    import yaml
    mace_config['train_file'] = inputs[0].filepath
    mace_config['valid_file'] = None
    config_str = yaml.dump(dict(mace_config))
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd  = 'cd $mytmpdir;'
    command_write = 'echo "{}" > config.yaml;'.format(config_str)
    command_list = [
            command_tmp,
            command_cd,
            command_write,
            'psiflow-train-mace',
            '--config=config.yaml',
            '--model=None',
            '--init_only;',
            'ls *;',
            'cp undeployed.pth {};'.format(outputs[0].filepath),
            'cp config.yaml {};'.format(outputs[1].filepath),
            #'touch {};'.format(outputs[0].filepath),
            #'touch {};'.format(outputs[1].filepath),
            ]
    return ' '.join(command_list)


def train(
        mace_config: dict,
        stdout: str = '',
        stderr: str = '',
        inputs: List[File] = [],
        outputs: List[File] = [],
        walltime: float = 1e12, # infinite by default
        parsl_resource_specification: dict = None,
        ) -> str:
    import yaml
    mace_config['train_file'] = inputs[1].filepath
    mace_config['valid_file'] = inputs[2].filepath
    config_str = yaml.dump(dict(mace_config))
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd  = 'cd $mytmpdir;'
    command_write = 'echo "{}" > config.yaml;'.format(config_str)
    command_list = [
            command_tmp,
            command_cd,
            command_write,
            'timeout -s 15 {}s psiflow-train-mace'.format(max(walltime - 15, 0)),
            '--config config.yaml',
            '--model {};'.format(inputs[0].filepath),
            'ls *;',
            'cp model/mace.model {};'.format(outputs[0].filepath), # no swa
            ]
    return ' '.join(command_list)


@typeguard.typechecked
def deploy(
        device: str,
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> None:
    import torch
    # model always stored on CPU after training
    model = torch.load(inputs[0].filepath, map_location='cpu')
    model.to(device=torch.device(device), dtype=torch.float32)
    torch.save(model, outputs[0].filepath)


@typeguard.typechecked
class MACEModel(BaseModel):

    def __init__(self, config: Union[dict, MACEConfig]) -> None:
        if isinstance(config, MACEConfig):
            config = asdict(config)
        else:
            config = dict(config)
        assert not config['swa'], 'usage of SWA is currently not supported'
        assert config['model_dtype'] == 'float32', 'dtype is enforced to float32'
        assert config['save_cpu'] # assert model is saved to CPU after training
        assert not 'hidden_irreps' in config.keys() # old MACE API
        config['device'] = 'cpu' # guarantee consistent initialization
        super().__init__(config)

    def deploy(self):
        self.deploy_future = psiflow.context().apps(self.__class__, 'deploy')(
                inputs=[self.model_future],
                outputs=[psiflow.context().new_file('deploy_', '.pth')],
                ).outputs[0]

    @classmethod
    def create_apps(cls) -> None:
        context = psiflow.context()
        evaluation, training = context[cls]
        training_label    = training.name()
        training_walltime = training.max_walltime
        training_ncores   = training.cores_per_worker
        if isinstance(get_active_executor(training_label), WorkQueueExecutor):
            training_resource_specification = training.generate_parsl_resource_specification()
        else:
            training_resource_specification = {}
        model_label  = evaluation.name()
        model_device = 'cuda' if evaluation.gpu else 'cpu'
        model_ncores = evaluation.cores_per_worker

        app_initialize = bash_app(initialize, executors=['Default'])
        def wrapped_deploy(inputs=[], outputs=[]):
            assert len(inputs) == 1
            assert len(outputs) == 1
            return deploy(model_device, inputs=inputs, outputs=outputs)
        app_deploy = python_app(wrapped_deploy, executors=[model_label])
        context.register_app(cls, 'deploy', app_deploy)
        def initialize_wrapped(config_raw, inputs=[], outputs=[]):
            assert len(inputs) == 1
            outputs = [
                    context.new_file('model_', '.pth'),
                    context.new_file('config_', '.yaml'),
                    ]
            init_future = app_initialize(
                    config_raw,
                    inputs=inputs,
                    outputs=outputs,
                    stdout=parsl.AUTO_LOGNAME,
                    stderr=parsl.AUTO_LOGNAME,
                    )
            future = read_yaml(inputs=[init_future.outputs[1]])
            deploy_future = app_deploy(
                    inputs=[init_future.outputs[0]],
                    outputs=[context.new_file('deploy_', '.pth')],
                    )
            return future, init_future.outputs[0], deploy_future.outputs[0]
        context.register_app(cls, 'initialize', initialize_wrapped)

        train_unwrapped = bash_app(train, executors=[training_label])
        def train_wrapped(config, inputs=[], outputs=[]):
            outputs = [context.new_file('model_', '.pth')]
            future = train_unwrapped(
                    config,
                    stdout=parsl.AUTO_LOGNAME,
                    stderr=parsl.AUTO_LOGNAME,
                    inputs=inputs,
                    outputs=outputs,
                    walltime=training_walltime * 60,
                    parsl_resource_specification=training_resource_specification,
                    )
            deploy_future = app_deploy(
                    inputs=[future.outputs[0]],
                    outputs=[context.new_file('deploy_', '.pth')],
                    )
            return future.outputs[0], deploy_future.outputs[0]
        context.register_app(cls, 'train', train_wrapped)
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
        context.register_app(cls, 'evaluate', evaluate_wrapped)

    @classmethod
    def load_calculator(
            cls,
            path_model: Union[Path, str],
            device: str,
            set_global_options: str = 'warn',
            ) -> BaseCalculator:
        from mace.calculators import MACECalculator
        return MACECalculator(
                model_paths=path_model,
                device=device,
                default_dtype='float32',
                )

    @property
    def seed(self) -> int:
        return self.config_raw['seed']

    @seed.setter
    def seed(self, arg: int) -> None:
        self.config_raw['seed'] = arg
