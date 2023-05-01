"""

NequIP train script -- copied from nequip@v0.5.6 with the following changes:

    - always use fresh_start() for training, with additional arguments to load
      an initialized config and model. The entry point is a now a new main()
      function which overrides the root directory in the config to the current
      working directory

    - fresh_start takes an additional argument path_model which is used to load
      a state dict of the model, to be stored in the trainer instance before
      training

    - ignore equivariance test, grad anomaly mode, model debug mode

"""


import logging
import argparse
import warnings

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from os.path import isdir
from pathlib import Path

import torch

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.utils import load_file
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.utils.versions import check_code_version
from nequip.utils._global_options import _set_global_options
from nequip.scripts._logger import set_up_script_logger

default_config = dict(
    root="./",
    run_name="NequIP",
    wandb=False,
    wandb_project="NequIP",
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "ForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    default_dtype="float32",
    allow_tf32=False,  # TODO: until we understand equivar issues
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    append=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy=[("DYNAMIC", 3)],
)


def main():
    """Entry point for the train bash_app of psiflow

    Should be executed in a temporary directory; this is set as nequip's root
    directory in which the results/wandb folder structure is created. After
    training, only the best model is retained

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to initialized config', default='', type=str)
    parser.add_argument('--model', help='path to undeployed model', default='', type=str)
    parser.add_argument('--ntrain', help='number of configurations in training set', default=0, type=int)
    parser.add_argument('--nvalid', help='number of configurations in validation set', default=0, type=int)
    args = parser.parse_args()

    # hacky! if remove 'energy' label when training on 'formation_energy'
    config = Config.from_file(args.config, defaults=default_config)
    if 'formation_energy' in config['dataset_key_mapping'].keys():
        from ase.io.extxyz import read_extxyz, write_extxyz
        with open(config['dataset_file_name'], 'r') as f:
            data = list(read_extxyz(f, index=slice(None)))
            for atoms in data:
                atoms.info['energy'] = atoms.info['formation_energy']
                atoms.calc = None
        with open(config['dataset_file_name'], 'w') as f:
            write_extxyz(f, data)
        with open(config['validation_dataset_file_name'], 'r') as f:
            data = list(read_extxyz(f, index=slice(None)))
            for atoms in data:
                atoms.info['energy'] = atoms.info['formation_energy']
                atoms.calc = None
        with open(config['validation_dataset_file_name'], 'w') as f:
            write_extxyz(f, data)

    import os
    config['root'] = os.getcwd()
    trainer = fresh_start(config, args.model)
    assert trainer.n_train == args.ntrain # should have been set in config by bash_app
    assert trainer.n_val   == args.nvalid
    trainer.save()
    trainer.train()


def fresh_start(config, path_model):
    # we use add_to_config cause it's a fresh start and need to record it
    check_code_version(config, add_to_config=True)
    _set_global_options(config)

    # = Make the trainer =
    if config.wandb:
        import wandb  # noqa: F401
        from nequip.train.trainer_wandb import TrainerWandB

        # download parameters from wandb in case of sweeping
        from psiflow.models._nequip import init_n_update

        config = init_n_update(config)

        trainer = TrainerWandB(model=None, **dict(config))
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer(model=None, **dict(config))

    # what is this
    # to update wandb data?
    config.update(trainer.params)

    # = Load the dataset =
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None

    # = Train/test split =
    trainer.set_dataset(dataset, validation_dataset)

    # = Build model =
    #final_model = model_from_config(
    #    config=config, initialize=True, dataset=trainer.dataset_train
    #)
    model = model_from_config(config, initialize=False)
    logging.info("Successfully built the network...")
    logging.info("psiflow-modification: loading state dict into model; "
            "setting precision float32; transferring to device cuda")
    model.load_state_dict(torch.load(path_model, map_location='cpu'))
    model.to(device=torch.device('cuda'), dtype=torch.float32)
    logging.info("psiflow-modification: success")

    # by doing this here we check also any keys custom builders may have added
    _check_old_keys(config)

    # Equivar test
    #if config.equivariance_test > 0:
    #    n_train: int = len(trainer.dataset_train)
    #    assert config.equivariance_test <= n_train
    #    final_model.eval()
    #    indexes = torch.randperm(n_train)[: config.equivariance_test]
    #    errstr = assert_AtomicData_equivariant(
    #        final_model, [trainer.dataset_train[i] for i in indexes]
    #    )
    #    final_model.train()
    #    logging.info(
    #        "Equivariance test passed; equivariance errors:\n"
    #        "   Errors are in real units, where relevant.\n"
    #        "   Please note that the large scale of the typical\n"
    #        "   shifts to the (atomic) energy can cause\n"
    #        "   catastrophic cancellation and give incorrectly\n"
    #        "   the equivariance error as zero for those fields.\n"
    #        f"{errstr}"
    #    )
    #    del errstr, indexes, n_train

    # Set the trainer
    trainer.model = model

    # Store any updated config information in the trainer
    trainer.update_kwargs(config)

    return trainer


def _check_old_keys(config) -> None:
    """check ``config`` for old/depricated keys and emit corresponding errors/warnings"""
    # compile_model
    k = "compile_model"
    if k in config:
        if config[k]:
            raise ValueError("the `compile_model` option has been removed")
        else:
            warnings.warn("the `compile_model` option has been removed")
