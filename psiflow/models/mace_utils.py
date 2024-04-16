###########################################################################################
# Modified training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs, Sander Vandenhaute
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

"""

MACE utils for use in psiflow -- initially copied from mace@d520aba
The following changes were made:

    - use signal module to wrap tools.train() call with timeout such that
    there is time left to save the best model.

    - build model from scratch but load state dict of starting model

    - simplified Calculator which incorporates additional atomic energy offsets

"""

import ast
import json
import logging
import os
import signal
from pathlib import Path
from typing import Optional

import mace
import numpy as np
import torch
import torch.nn.functional
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.scripts_utils import (
    LRScheduler,
    create_error_table,
    get_dataset_from_xyz,
)
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage

from psiflow.hamiltonians.utils import check_forces


class MACECalculator(Calculator):
    """MACE ASE Calculator"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: str,
        atomic_energies: dict[float, str],
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        max_force: Optional[float] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        torch_tools.set_default_dtype(dtype)
        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.model = torch.load(f=model_path, map_location=device)
        if dtype == "float64":
            self.model = self.model.double()
        self.model.to(device)
        self.model.eval()
        self.r_max = float(self.model.r_max)
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )

        # store atomic energies from psiflow in calculator
        self.atomic_energies = atomic_energies
        self.max_force = max_force

    def get_atomic_energy(self, atoms):
        total = 0
        natoms = len(atoms)
        for symbol in list(set(atoms.symbols)):
            natoms_per_symbol = np.sum(np.array(atoms.symbols) == symbol)
            if natoms_per_symbol == 0:
                continue
            total += natoms_per_symbol * self.atomic_energies[symbol]
            natoms -= natoms_per_symbol
        assert natoms == 0  # all atoms accounted for
        return total

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # check that if atomic energies dict is non-empty, all chemical elements are present
        symbols = list(set(atoms.symbols))
        if len(self.atomic_energies) > 0:
            assert np.all([s in self.atomic_energies for s in symbols])

        # prepare data
        config = data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)

        # predict + extract data
        out = self.model(batch.to_dict(), compute_stress=True)
        energy = out["energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()

        # store results
        E = energy * self.energy_units_to_eV

        # add atomic energies if necessary
        if len(self.atomic_energies) > 0:
            atomic_energy = self.get_atomic_energy(atoms)
            E += atomic_energy * self.energy_units_to_eV

        if self.max_force is not None:
            check_forces(
                forces,
                atoms,
                self.max_force,
            )
        self.results = {
            "energy": E,
            "free_energy": E,
            # force has units eng / len:
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
        }

        # even though compute_stress is True, stress can be none if pbc is False
        # not sure if correct ASE thing is to have no dict key, or dict key with value None
        if out["stress"] is not None:
            stress = out["stress"].detach().cpu().numpy()
            # stress has units eng / len^3:
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def main() -> None:
    # extend MACE arg parser with ability to pass initialized model; set tmpdirs
    parser = tools.build_default_arg_parser()
    parser.add_argument(
        "--initialized_model",
        help="path to initialized model",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    args.log_dir = os.path.join(os.getcwd(), "log")
    args.model_dir = os.path.join(os.getcwd())
    args.results_dir = os.path.join(os.getcwd(), "results")
    args.downloads_dir = os.path.join(os.getcwd(), "downloads")
    args.checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")

    tag = tools.get_tag(name=args.name, seed=args.seed)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {args}")
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        charges_key=args.charges_key,
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
    if args.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        compute_dipole = True
        compute_energy = False
        args.compute_forces = False
        compute_virials = False
        args.compute_stress = False
    else:
        dipole_only = False
        if args.model == "EnergyDipolesMACE":
            compute_dipole = True
            compute_energy = True
            args.compute_forces = True
            compute_virials = False
            args.compute_stress = False
        else:
            compute_energy = True
            compute_dipole = False
        if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
            if args.E0s is not None:
                logging.info(
                    "Atomic Energies not in training file, using command line argument E0s"
                )
                if args.E0s.lower() == "average":
                    logging.info(
                        "Computing average Atomic Energies using least squares regression"
                    )
                    atomic_energies_dict = data.compute_average_E0s(
                        collections.train, z_table
                    )
                else:
                    try:
                        atomic_energies_dict = ast.literal_eval(args.E0s)
                        assert isinstance(atomic_energies_dict, dict)
                    except Exception as e:
                        raise RuntimeError(
                            f"E0s specified invalidly, error {e} occurred"
                        ) from e
            else:
                raise RuntimeError(
                    "E0s not found in training file and not specified in command line"
                )
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")
    args.batch_size = min(len(collections.train), args.batch_size)
    print("actual batch size: {}".format(args.batch_size))

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.train
        ],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.valid
        ],
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
    )

    loss_fn: torch.nn.Module
    if args.loss == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    elif args.loss == "forces_only":
        loss_fn = modules.WeightedForcesLoss(forces_weight=args.forces_weight)
    elif args.loss == "virials":
        loss_fn = modules.WeightedEnergyForcesVirialsLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            virials_weight=args.virials_weight,
        )
    elif args.loss == "stress":
        loss_fn = modules.WeightedEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
        )
    elif args.loss == "huber":
        loss_fn = modules.WeightedHuberEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
            huber_delta=args.huber_delta,
        )
    elif args.loss == "dipole":
        assert (
            dipole_only is True
        ), "dipole loss can only be used with AtomicDipolesMACE model"
        loss_fn = modules.DipoleSingleLoss(
            dipole_weight=args.dipole_weight,
        )
    elif args.loss == "energy_forces_dipole":
        assert dipole_only is False and compute_dipole is True
        loss_fn = modules.WeightedEnergyForcesDipoleLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            dipole_weight=args.dipole_weight,
        )
    else:
        # Unweighted Energy and Forces loss by default
        loss_fn = modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    logging.info(loss_fn)

    if args.compute_avg_num_neighbors:
        args.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {args.avg_num_neighbors}")

    # Selecting outputs
    compute_virials = False
    if args.loss in ("stress", "virials", "huber"):
        compute_virials = True
        args.compute_stress = True
        args.error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": compute_dipole,
    }
    logging.info(f"Selected the following outputs: {output_args}")

    # Build model
    logging.info("Building model")
    if args.num_channels is not None and args.max_L is not None:
        assert args.num_channels > 0, "num_channels must be positive integer"
        assert args.max_L >= 0, "max_L must be non-negative integer"
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )

    assert (
        len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
    ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

    logging.info(f"Hidden irreps: {args.hidden_irreps}")
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
    )

    model: torch.nn.Module

    if args.model == "MACE":
        if args.scaling == "no_scaling":
            std = 1.0
            logging.info("No scaling selected")
        else:
            mean, std = modules.scaling_classes[args.scaling](
                train_loader, atomic_energies
            )
        model = modules.ScaleShiftMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
        )
    elif args.model == "ScaleShiftMACE":
        mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
        model = modules.ScaleShiftMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
        )
    elif args.model == "ScaleShiftBOTNet":
        mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
        model = modules.ScaleShiftBOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif args.model == "BOTNet":
        model = modules.BOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    elif args.model == "AtomicDipolesMACE":
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            args.error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        model = modules.AtomicDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            # dipole_scale=1,
            # dipole_shift=0,
        )
    elif args.model == "EnergyDipolesMACE":
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert (
            args.loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            args.error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        model = modules.EnergyDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    else:
        raise RuntimeError(f"Unknown model: '{args.model}'")

    if args.initialized_model is None:  # save currently initialized model
        torch.save(model.to("cpu"), "model.pth")
        return 0
    else:  # override model with initialized state_dict
        state_dict = torch.load(args.initialized_model, map_location="cpu").state_dict()
        model.load_state_dict(state_dict)

    model = model.to(device)

    # Optimizer
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
                "weight_decay": args.weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
        amsgrad=args.amsgrad,
    )

    optimizer: torch.optim.Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        assert dipole_only is False, "swa for dipole fitting not implemented"
        swas.append(True)
        if args.start_swa is None:
            args.start_swa = (
                args.max_num_epochs // 4 * 3
            )  # if not set start swa at 75% of training
        if args.loss == "forces_only":
            logging.info("Can not select swa with forces only loss.")
        elif args.loss == "virials":
            loss_fn_energy = modules.WeightedEnergyForcesVirialsLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                virials_weight=args.swa_virials_weight,
            )
        elif args.loss == "stress":
            loss_fn_energy = modules.WeightedEnergyForcesStressLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                stress_weight=args.swa_stress_weight,
            )
        elif args.loss == "energy_forces_dipole":
            loss_fn_energy = modules.WeightedEnergyForcesDipoleLoss(
                args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                dipole_weight=args.swa_dipole_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
            )
        else:
            loss_fn_energy = modules.WeightedEnergyForcesLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight} and learning rate : {args.swa_lr}"
            )
        swa = tools.SWAContainer(
            model=AveragedModel(model),
            scheduler=SWALR(
                optimizer=optimizer,
                swa_lr=args.swa_lr,
                anneal_epochs=1,
                anneal_strategy="linear",
            ),
            start=args.start_swa,
            loss_fn=loss_fn_energy,
        )

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception:  # pylint: disable=W0703
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    logging.info(model)
    logging.info(f"Number of parameters: {tools.count_parameters(model)}")
    logging.info(f"Optimizer: {optimizer}")

    if args.wandb:
        logging.info("Using Weights and Biases for logging")
        import wandb

        wandb_config = {}
        args_dict = vars(args)
        args_dict_json = json.dumps(args_dict)
        for key in args.wandb_log_hypers:
            wandb_config[key] = args_dict[key]
        tools.init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=wandb_config,
        )
        wandb.run.summary["params"] = args_dict_json

    try:
        tools.train(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_handler=checkpoint_handler,
            eval_interval=args.eval_interval,
            start_epoch=start_epoch,
            max_num_epochs=args.max_num_epochs,
            logger=logger,
            patience=args.patience,
            output_args=output_args,
            device=device,
            swa=swa,
            ema=ema,
            max_grad_norm=args.clip_grad,
            log_errors=args.error_table,
            log_wandb=args.wandb,
        )
    except TimeoutException:
        logging.info("received SIGTERM!")
        pass

    # Evaluation on test datasets
    logging.info("Computing metrics for training, validation, and test sets")

    all_collections = [
        ("train", collections.train),
        ("valid", collections.valid),
    ] + collections.tests

    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=swa_eval,
            device=device,
        )
        model.to(device)
        logging.info(f"Loaded model from epoch {epoch}")

        for param in model.parameters():
            param.requires_grad = False
        table = create_error_table(
            table_type=args.error_table,
            all_collections=all_collections,
            z_table=z_table,
            r_max=args.r_max,
            valid_batch_size=args.valid_batch_size,
            model=model,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
        )
        logging.info("\n" + str(table))

        # Save entire model
        if swa_eval:
            model_path = Path(args.checkpoints_dir) / (tag + "_swa.model")
        else:
            model_path = Path(args.checkpoints_dir) / (tag + ".model")
        logging.info(f"Saving model to {model_path}")
        if args.save_cpu:
            model = model.to("cpu")
        torch.save(model, "model.pth")

        if swa_eval:
            torch.save(model, Path(args.model_dir) / (args.name + "_swa.model"))
        else:
            torch.save(model, Path(args.model_dir) / (args.name + ".model"))

    logging.info("Done")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, timeout_handler)
    main()
