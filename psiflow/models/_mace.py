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
from psiflow.utils import get_active_executor
from psiflow.execution import ExecutionContext, ModelTrainingExecution, \
        ModelEvaluationExecution


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
    interaction: str = 'RealAgnosticResidualInteractionBlock'
    interaction_first: str = 'RealAgnosticResidualInteractionBlock'
    max_ell: int = 3
    correlation: int = 3
    num_interactions: int = 2
    MLP_irreps: str = '16x0e'
    hidden_irreps: str = '32x0e'
    gate: str = 'silu'
    scaling: str = 'rms_forces_scaling'
    avg_num_neighbors: Optional[float] = None
    compute_avg_num_neighbors: bool = True
    compute_stress: bool = True
    compute_forces: bool = True
    device: str = 'cuda'
    default_dtype: str = 'float32'
    config_type_weights: str = '{"Default":1.0}'
    valid_fraction: float = 0.1
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
    swa_energy_weight: float = 10
    stress_weight: float = 10
    swa_stress_weight: float = 10
    optimizer: str = 'adam'
    batch_size: int = 1
    valid_batch_size: int = 1
    lr: float = 0.01
    swa_lr: float = 0.001
    weight_decay: float = 1e-7
    amsgrad: bool = True
    scheduler: str = 'ReduceLROnPlateau'
    lr_factor: float = 0.8
    scheduler_patience: int = 100
    lr_scheduler_gamma: float = 0.9993
    swa: bool = False
    start_swa: Optional[int] = None
    ema: bool = False
    ema_decay: float = 0.99
    max_num_epochs: int = int(1e6)
    patience: int = 2048
    eval_interval: int = 2
    keep_checkpoints: bool = False
    restart_latest: bool = False
    save_cpu: bool = False
    clip_grad: Optional[float] = 10


@typeguard.typechecked
def initialize( # taken from MACE @ d520aba
        config_dict: dict,
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> dict:
    import torch
    import ast
    #import logging
    from pathlib import Path
    from typing import Optional
    import tempfile

    import numpy as np
    from e3nn import o3

    import mace
    from mace import data, modules, tools
    from mace.tools import torch_geometric
    from mace.tools.scripts_utils import get_dataset_from_xyz

    from psiflow.models import MACEConfig

    mace_config = MACEConfig(**config_dict)
    mace_config.log_dir = tempfile.mkdtemp()
    mace_config.model_dir = tempfile.mkdtemp()
    mace_config.results_dir = tempfile.mkdtemp()
    mace_config.downloads_dir = tempfile.mkdtemp()
    mace_config.checkpoints_dir = tempfile.mkdtemp()
    tag = tools.get_tag(name=mace_config.name, seed=mace_config.seed)

    # Setup
    tools.set_seeds(mace_config.seed)
    #tools.setup_logger(level=mace_config.log_level, tag=tag, directory=mace_config.log_dir)
    #try:
    #    logging.info(f"MACE version: {mace.__version__}")
    #except AttributeError:
    #    logging.info("Cannot find MACE version, please install MACE via pip")
    #logging.info(f"Configuration: {mace_config}")
    device = tools.init_device(mace_config.device)
    tools.set_default_dtype(mace_config.default_dtype)

    try:
        config_type_weights = ast.literal_eval(mace_config.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        #logging.warning(
        #    f"Config type weights not specified correctly ({e}), using Default"
        #)
        config_type_weights = {"Default": 1.0}

    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=inputs[0].filepath,
        #valid_path=config.valid_file,
        valid_path=None,
        valid_fraction=mace_config.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=mace_config.test_file,
        seed=mace_config.seed,
        energy_key=mace_config.energy_key,
        forces_key=mace_config.forces_key,
        stress_key=mace_config.stress_key,
        virials_key=mace_config.virials_key,
        dipole_key=mace_config.dipole_key,
        charges_key=mace_config.charges_key,
    )

    #logging.info(
    #    f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
    #    f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
    #)

    # Atomic number table
    # yapf: disable
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    # yapf: enable
    #logging.info(z_table)
    if mace_config.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        compute_dipole = True
        compute_energy = False
        mace_config.compute_forces = False
        compute_virials = False
        mace_config.compute_stress = False
    else:
        dipole_only = False
        if mace_config.model == "EnergyDipolesMACE":
            compute_dipole = True
            compute_energy = True
            mace_config.compute_forces = True
            compute_virials = False
            mace_config.compute_stress = False
        else:
            compute_energy = True
            compute_dipole = False
        if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
            if mace_config.E0s is not None:
                #logging.info(
                #    "Atomic Energies not in training file, using command line argument E0s"
                #)
                if mace_config.E0s.lower() == "average":
                    #logging.info(
                    #    "Computing average Atomic Energies using least squares regression"
                    #)
                    atomic_energies_dict = data.compute_average_E0s(
                        collections.train, z_table
                    )
                else:
                    try:
                        atomic_energies_dict = ast.literal_eval(mace_config.E0s)
                        assert isinstance(atomic_energies_dict, dict)
                    except Exception as e:
                        raise RuntimeError(
                            f"E0s specified invalidly, error {e} occured"
                        ) from e
            else:
                raise RuntimeError(
                    "E0s not found in training file and not specified in command line"
                )
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        #logging.info(f"Atomic energies: {atomic_energies.tolist()}")

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=mace_config.r_max)
            for config in collections.train
        ],
        batch_size=mace_config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=mace_config.r_max)
            for config in collections.valid
        ],
        batch_size=mace_config.valid_batch_size,
        shuffle=False,
        drop_last=False,
    )

    loss_fn: torch.nn.Module
    if mace_config.loss == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=mace_config.energy_weight, forces_weight=mace_config.forces_weight
        )
    elif mace_config.loss == "forces_only":
        loss_fn = modules.WeightedForcesLoss(forces_weight=mace_config.forces_weight)
    elif mace_config.loss == "virials":
        loss_fn = modules.WeightedEnergyForcesVirialsLoss(
            energy_weight=mace_config.energy_weight,
            forces_weight=mace_config.forces_weight,
            virials_weight=mace_config.virials_weight,
        )
    elif mace_config.loss == "stress":
        loss_fn = modules.WeightedEnergyForcesStressLoss(
            energy_weight=mace_config.energy_weight,
            forces_weight=mace_config.forces_weight,
            stress_weight=mace_config.stress_weight,
        )
    elif mace_config.loss == "dipole":
        assert (
            dipole_only is True
        ), "dipole loss can only be used with AtomicDipolesMACE model"
        loss_fn = modules.DipoleSingleLoss(
            dipole_weight=mace_config.dipole_weight,
        )
    elif mace_config.loss == "energy_forces_dipole":
        assert dipole_only is False and compute_dipole is True
        loss_fn = modules.WeightedEnergyForcesDipoleLoss(
            energy_weight=mace_config.energy_weight,
            forces_weight=mace_config.forces_weight,
            dipole_weight=mace_config.dipole_weight,
        )
    else:
        loss_fn = modules.EnergyForcesLoss(
            energy_weight=mace_config.energy_weight, forces_weight=mace_config.forces_weight
        )
    #logging.info(loss_fn)

    if mace_config.compute_avg_num_neighbors:
        mace_config.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
    #logging.info(f"Average number of neighbors: {mace_config.avg_num_neighbors:.3f}")

    # Selecting outputs
    compute_virials = False
    if mace_config.loss in ("stress", "virials"):
        compute_virials = True
        mace_config.compute_stress = True
        mace_config.error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": compute_energy,
        "forces": mace_config.compute_forces,
        "virials": compute_virials,
        "stress": mace_config.compute_stress,
        "dipoles": compute_dipole,
    }
    #logging.info(f"Selected the following outputs: {output_args}")

    # Build model
    #logging.info("Building model")
    model_config = dict(
        r_max=mace_config.r_max,
        num_bessel=mace_config.num_radial_basis,
        num_polynomial_cutoff=mace_config.num_cutoff_basis,
        max_ell=mace_config.max_ell,
        interaction_cls=modules.interaction_classes[mace_config.interaction],
        num_interactions=mace_config.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(mace_config.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=mace_config.avg_num_neighbors,
        atomic_numbers=z_table.zs,
    )

    model: torch.nn.Module

    if mace_config.model == "MACE":
        if mace_config.scaling == "no_scaling":
            std = 1.0
            #logging.info("No scaling selected")
        else:
            mean, std = modules.scaling_classes[mace_config.scaling](
                train_loader, atomic_energies
            )
        model = modules.ScaleShiftMACE(
            **model_config,
            correlation=mace_config.correlation,
            gate=modules.gate_dict[mace_config.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(mace_config.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
        )
    elif mace_config.model == "ScaleShiftMACE":
        mean, std = modules.scaling_classes[mace_config.scaling](train_loader, atomic_energies)
        model = modules.ScaleShiftMACE(
            **model_config,
            correlation=mace_config.correlation,
            gate=modules.gate_dict[mace_config.gate],
            interaction_cls_first=modules.interaction_classes[mace_config.interaction_first],
            MLP_irreps=o3.Irreps(mace_config.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif mace_config.model == "ScaleShiftBOTNet":
        mean, std = modules.scaling_classes[mace_config.scaling](train_loader, atomic_energies)
        model = modules.ScaleShiftBOTNet(
            **model_config,
            gate=modules.gate_dict[mace_config.gate],
            interaction_cls_first=modules.interaction_classes[mace_config.interaction_first],
            MLP_irreps=o3.Irreps(mace_config.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif mace_config.model == "BOTNet":
        model = modules.BOTNet(
            **model_config,
            gate=modules.gate_dict[mace_config.gate],
            interaction_cls_first=modules.interaction_classes[mace_config.interaction_first],
            MLP_irreps=o3.Irreps(mace_config.MLP_irreps),
        )
    elif mace_config.model == "AtomicDipolesMACE":
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert mace_config.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            mace_config.error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        model = modules.AtomicDipolesMACE(
            **model_config,
            correlation=mace_config.correlation,
            gate=modules.gate_dict[mace_config.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(mace_config.MLP_irreps),
            # dipole_scale=1,
            # dipole_shift=0,
        )
    elif mace_config.model == "EnergyDipolesMACE":
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert (
            mace_config.loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            mace_config.error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        model = modules.EnergyDipolesMACE(
            **model_config,
            correlation=mace_config.correlation,
            gate=modules.gate_dict[mace_config.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(mace_config.MLP_irreps),
        )
    else:
        raise RuntimeError(f"Unknown model: '{mace_config.model}'")
    model.to(device)
    torch.save(model, outputs[0].filepath)

    # set computed properties in config and remove temp folders
    mace_config.compute_avg_num_neighbors = False # value already set
    #E0s = ['{}:{}'.format(z, e) for z, e in zip(atomic_energies_dict.keys(), atomic_energies_dict.values())]
    mace_config.E0s = str(atomic_energies_dict)
    mace_config.log_dir = ''
    mace_config.model_dir = ''
    mace_config.results_dir = ''
    mace_config.downloads_dir = ''
    mace_config.checkpoints_dir = ''
    return asdict(mace_config)


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
            'psiflow-train-mace',
            '--config config.yaml',
            '--time {}'.format(max(walltime - 100, 0)), # 100 s slack
            '--model {};'.format(inputs[0].filepath),
            'ls *;',
            'cp model/mace.model {};'.format(outputs[0].filepath), # no swa
            ]
    return ' '.join(command_list)


@typeguard.typechecked
def deploy(
        device: str,
        dtype: str,
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> None:
    import torch
    model = torch.load(inputs[0].filepath, map_location='cpu')
    torch_device = torch.device(device)
    if dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float64
    model.to(device=torch_device, dtype=torch_dtype)
    torch.save(model, outputs[0].filepath)


@typeguard.typechecked
class MACEModel(BaseModel):

    def __init__(self, config: Union[dict, MACEConfig]) -> None:
        if isinstance(config, MACEConfig):
            config = asdict(config)
        else:
            config = dict(config)
        assert not config['swa'], 'usage of SWA is currently not supported'
        config['device'] = 'cpu' # guarantee consistent initialization
        super().__init__(config)

    def deploy(self) -> None:
        assert self.config_future is not None
        assert self.model_future is not None
        context = psiflow.context()
        self.deploy_future['float32'] = context.apps(MACEModel, 'deploy_float32')(
                inputs=[self.model_future],
                outputs=[context.new_file('deployed_', '.pth')],
                ).outputs[0]
        self.deploy_future['float64'] = context.apps(MACEModel, 'deploy_float64')(
                inputs=[self.model_future],
                outputs=[context.new_file('deployed_', '.pth')],
                ).outputs[0]

    def set_seed(self, seed: int) -> None:
        self.config_raw['seed'] = seed

    @classmethod
    def create_apps(cls) -> None:
        context = psiflow.context()
        for execution in context[cls]:
            if type(execution) == ModelTrainingExecution:
                training_label    = execution.executor
                training_walltime = execution.walltime
                training_ncores   = execution.ncores
                if isinstance(get_active_executor(training_label), WorkQueueExecutor):
                    training_resource_specification = execution.generate_parsl_resource_specification()
                else:
                    training_resource_specification = {}
            elif type(execution) == ModelEvaluationExecution:
                model_label    = execution.executor
                model_device   = execution.device
                model_dtype    = execution.dtype
                model_ncores   = execution.ncores

        app_initialize = python_app(initialize, executors=[model_label], cache=False)
        context.register_app(cls, 'initialize', app_initialize)
        deploy_unwrapped = python_app(deploy, executors=[model_label], cache=False)
        def deploy_float32(inputs=[], outputs=[]):
            return deploy_unwrapped(
                    model_device,
                    'float32',
                    inputs=inputs,
                    outputs=outputs,
                    )
        context.register_app(cls, 'deploy_float32', deploy_float32)
        def deploy_float64(inputs=[], outputs=[]):
            return deploy_unwrapped(
                    model_device,
                    'float64',
                    inputs=inputs,
                    outputs=outputs,
                    )
        context.register_app(cls, 'deploy_float64', deploy_float64)
        train_unwrapped = bash_app(train, executors=[training_label], cache=False)
        def train_wrapped(config, inputs=[], outputs=[]):
            return train_unwrapped(
                    config,
                    stdout=parsl.AUTO_LOGNAME,
                    stderr=parsl.AUTO_LOGNAME,
                    inputs=inputs,
                    outputs=outputs,
                    walltime=training_walltime * 60,
                    parsl_resource_specification=training_resource_specification,
                    )
        context.register_app(cls, 'train', train_wrapped)
        evaluate_unwrapped = python_app(
                evaluate_dataset,
                executors=[model_label],
                cache=False,
                )
        def evaluate_wrapped(deploy_future, use_formation_energy, inputs=[], outputs=[]):
            assert model_dtype in deploy_future.keys(), ('model is not '
                    'deployed; use model.deploy() before using model.evaluate()')
            inputs.append(deploy_future[model_dtype])
            return evaluate_unwrapped(
                    model_device,
                    model_dtype,
                    model_ncores,
                    use_formation_energy,
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
            dtype: str,
            set_global_options: str = 'warn',
            ) -> BaseCalculator:
        from mace.calculators import MACECalculator
        return MACECalculator(
                model_path=path_model,
                device=device,
                default_dtype=dtype,
                )

    @property
    def use_formation_energy(self) -> bool:
        return self.config_raw['energy_key'] == 'formation_energy'

    @use_formation_energy.setter
    def use_formation_energy(self, arg) -> None:
        assert self.model_future is None
        if arg: # use formation_energy
            self.config_raw['energy_key'] = 'formation_energy'
        else: # switch to total energy
            self.config_raw['energy_key'] = 'energy'
