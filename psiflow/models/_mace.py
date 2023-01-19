from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Any, Dict
import typeguard
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

from ase.calculators.calculator import BaseCalculator

import parsl
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.models import BaseModel
from psiflow.models.base import evaluate_dataset
from psiflow.data import Dataset
from psiflow.execution import ExecutionContext


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


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
    config_type_weights: dict['str', float] = field(default_factory=lambda: {'Default': 1.0})
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
    lr_factor: str = 0.8
    scheduler_patience: int = 50
    lr_scheduler_gamma: float = 0.9993
    swa: bool = False
    start_swa: Optional[int] = None
    ema: bool = False
    ema_decay: float = 0.99
    max_num_epochs: int = 2048
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
    import logging
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
    tools.setup_logger(level=mace_config.log_level, tag=tag, directory=mace_config.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {mace_config}")
    device = tools.init_device(mace_config.device)
    tools.set_default_dtype(mace_config.default_dtype)

    try:
        config_type_weights = ast.literal_eval(mace_config.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
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

    logging.info(
        f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
        f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
    )

    # Atomic number table
    # yapf: disable
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    # yapf: enable
    logging.info(z_table)
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
                logging.info(
                    "Atomic Energies not in training file, using command line argument E0s"
                )
                if mace_config.E0s.lower() == "average":
                    logging.info(
                        "Computing average Atomic Energies using least squares regression"
                    )
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
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")

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
    logging.info(loss_fn)

    if mace_config.compute_avg_num_neighbors:
        mace_config.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {mace_config.avg_num_neighbors:.3f}")

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
    logging.info(f"Selected the following outputs: {output_args}")

    # Build model
    logging.info("Building model")
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
            logging.info("No scaling selected")
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


@typeguard.typechecked
def train(
        device: str,
        dtype: str,
        mace_config: dict,
        inputs: List[File] = [],
        outputs: List[File] = [],
        walltime: float = 3600,
        ) -> int:
    import ast
    import tempfile
    import torch
    import numpy as np
    from torch.optim.swa_utils import SWALR, AveragedModel
    from torch_ema import ExponentialMovingAverage
    import mace
    from mace import data, modules, tools
    from mace.tools import torch_geometric
    from mace.tools.scripts_utils import create_error_table, \
            get_dataset_from_xyz

    from psiflow.models import MACEConfig
    # override device and dtype in mace config
    mace_config = MACEConfig(**mace_config)
    mace_config.device = device
    mace_config.default_dtype = dtype

    mace_config.log_dir = tempfile.mkdtemp()
    mace_config.model_dir = tempfile.mkdtemp()
    mace_config.results_dir = tempfile.mkdtemp()
    mace_config.downloads_dir = tempfile.mkdtemp()
    mace_config.checkpoints_dir = tempfile.mkdtemp()
    tag = tools.get_tag(name=mace_config.name, seed=mace_config.seed)

    # Setup
    tools.set_seeds(mace_config.seed)
    tools.setup_logger(level=mace_config.log_level, tag=tag, directory=mace_config.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {mace_config}")
    device = tools.init_device(mace_config.device)
    tools.set_default_dtype(mace_config.default_dtype)

    try:
        config_type_weights = ast.literal_eval(mace_config.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=inputs[1].filepath,
        valid_path=inputs[2].filepath,
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

    logging.info(
        f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
        f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
    )

    # Atomic number table
    # yapf: disable
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    # yapf: enable
    logging.info(z_table)
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
                logging.info(
                    "Atomic Energies not in training file, using command line argument E0s"
                )
                if mace_config.E0s.lower() == "average":
                    logging.info(
                        "Computing average Atomic Energies using least squares regression"
                    )
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
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")

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
    logging.info(loss_fn)

    #if mace_config.compute_avg_num_neighbors:
    #    mace_config.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
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
    logging.info(f"Selected the following outputs: {output_args}")

    model = torch.load(inputs[0].filepath, map_location='cpu')
    torch_device = torch.device(device)
    if dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float64
    model.to(device=torch_device, dtype=torch_dtype)

    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": mace_config.weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": mace_config.weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=mace_config.lr,
        amsgrad=mace_config.amsgrad,
    )

    optimizer: torch.optim.Optimizer
    if mace_config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    logger = tools.MetricsLogger(directory=mace_config.results_dir, tag=tag + "_train")

    if mace_config.scheduler == "ExponentialLR":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=mace_config.lr_scheduler_gamma
        )
    elif mace_config.scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=mace_config.lr_factor,
            patience=mace_config.scheduler_patience,
        )
    else:
        raise RuntimeError(f"Unknown scheduler: '{mace_config.scheduler}'")

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if mace_config.swa:
        assert dipole_only is False, "swa for dipole fitting not implemented"
        swas.append(True)
        if mace_config.start_swa is None:
            mace_config.start_swa = (
                mace_config.max_num_epochs // 4 * 3
            )  # if not set start swa at 75% of training
        if mace_config.loss == "forces_only":
            logging.info("Can not select swa with forces only loss.")
        elif mace_config.loss == "virials":
            loss_fn_energy = modules.WeightedEnergyForcesVirialsLoss(
                energy_weight=mace_config.swa_energy_weight,
                forces_weight=mace_config.swa_forces_weight,
                virials_weight=mace_config.swa_virials_weight,
            )
        elif mace_config.loss == "stress":
            loss_fn_energy = modules.WeightedEnergyForcesStressLoss(
                energy_weight=mace_config.swa_energy_weight,
                forces_weight=mace_config.swa_forces_weight,
                stress_weight=mace_config.swa_stress_weight,
            )
        elif mace_config.loss == "energy_forces_dipole":
            loss_fn_energy = modules.WeightedEnergyForcesDipoleLoss(
                mace_config.swa_energy_weight,
                forces_weight=mace_config.swa_forces_weight,
                dipole_weight=mace_config.swa_dipole_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {mace_config.start_swa} epochs) with energy weight : {mace_config.swa_energy_weight}, forces weight : {mace_config.swa_forces_weight}, dipole weight : {mace_config.swa_dipole_weight} and learning rate : {mace_config.swa_lr}"
            )
        else:
            loss_fn_energy = modules.WeightedEnergyForcesLoss(
                energy_weight=mace_config.swa_energy_weight,
                forces_weight=mace_config.swa_forces_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {mace_config.start_swa} epochs) with energy weight : {mace_config.swa_energy_weight}, forces weight : {mace_config.swa_forces_weight} and learning rate : {mace_config.swa_lr}"
            )
        swa = tools.SWAContainer(
            model=AveragedModel(model),
            scheduler=SWALR(
                optimizer=optimizer,
                swa_lr=mace_config.swa_lr,
                anneal_epochs=1,
                anneal_strategy="linear",
            ),
            start=mace_config.start_swa,
            loss_fn=loss_fn_energy,
        )

    checkpoint_handler = tools.CheckpointHandler(
        directory=mace_config.checkpoints_dir,
        tag=tag,
        keep=mace_config.keep_checkpoints,
        swa_start=mace_config.start_swa,
    )

    start_epoch = 0
    if mace_config.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if mace_config.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=mace_config.ema_decay)

    logging.info(model)
    logging.info(f"Number of parameters: {tools.count_parameters(model)}")
    logging.info(f"Optimizer: {optimizer}")

    try:
        tools.train(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_handler=checkpoint_handler,
            eval_interval=mace_config.eval_interval,
            start_epoch=start_epoch,
            max_num_epochs=mace_config.max_num_epochs,
            logger=logger,
            patience=mace_config.patience,
            output_args=output_args,
            device=device,
            swa=swa,
            ema=ema,
            max_grad_norm=mace_config.clip_grad,
            log_errors=mace_config.error_table,
        )
        logging.info("Done")
    except parsl.app.errors.AppTimeout as e:
        pass
    epoch = checkpoint_handler.load_latest(
        state=tools.CheckpointState(model, optimizer, lr_scheduler), device=device
    )
    torch.save(model.to('cpu'), outputs[0].filepath)
    return epoch


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

    def __init__(
            self,
            context: ExecutionContext,
            config: Union[dict, MACEConfig],
            ) -> None:
        if isinstance(config, MACEConfig):
            config = asdict(config)
        else:
            config = dict(config)
        config['device'] = 'cpu' # guarantee consistent initialization
        super().__init__(context, config)

    def initialize(self, dataset: Dataset) -> None:
        assert self.config_future is None
        assert self.model_future is None
        self.deploy_future = {}
        logger.info('initializing {} using dataset of {} states'.format(
            self.__class__.__name__, dataset.length().result()))
        self.config_future = self.context.apps(self.__class__, 'initialize')(
                self.config_raw,
                inputs=[dataset.data_future],
                outputs=[self.context.new_file('model_', '.pth')],
                )
        self.model_future = self.config_future.outputs[0] # to undeployed model

    def deploy(self) -> None:
        assert self.config_future is not None
        assert self.model_future is not None
        self.deploy_future['float32'] = self.context.apps(MACEModel, 'deploy_float32')(
                inputs=[self.model_future],
                outputs=[self.context.new_file('deployed_', '.pth')],
                ).outputs[0]
        self.deploy_future['float64'] = self.context.apps(MACEModel, 'deploy_float64')(
                inputs=[self.model_future],
                outputs=[self.context.new_file('deployed_', '.pth')],
                ).outputs[0]

    def set_seed(self, seed: int) -> None:
        self.config_raw['seed'] = seed

    @classmethod
    def create_apps(cls, context: ExecutionContext) -> None:
        training_label    = context[cls]['training_executor']
        training_device   = context[cls]['training_device']
        training_dtype    = context[cls]['training_dtype']
        training_walltime = context[cls]['training_walltime']

        model_label  = context[cls]['evaluate_executor']
        model_device = context[cls]['evaluate_device']
        model_ncores = context[cls]['evaluate_ncores']
        model_dtype  = context[cls]['evaluate_dtype']

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
        train_unwrapped = python_app(train, executors=[training_label], cache=False)
        def train_wrapped(config, inputs=[], outputs=[]):
            return train_unwrapped(
                    training_device,
                    training_dtype,
                    config,
                    inputs=inputs,
                    outputs=outputs,
                    walltime=training_walltime,
                    )
        context.register_app(cls, 'train', train_wrapped)
        evaluate_unwrapped = python_app(
                evaluate_dataset,
                executors=[model_label],
                cache=False,
                )
        def evaluate_wrapped(inputs=[], outputs=[]):
            return evaluate_unwrapped(
                    model_device,
                    model_dtype,
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
            dtype: str,
            set_global_options: str = 'warn',
            ) -> BaseCalculator:
        from mace.calculators import MACECalculator
        return MACECalculator(
                model_path=path_model,
                device=device,
                default_dtype=dtype,
                )
