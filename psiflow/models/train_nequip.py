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


import argparse
import logging
import warnings
from pathlib import Path

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401
import torch
import yaml
from ase.io import read, write
from nequip.data import dataset_from_config
from nequip.model import model_from_config
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options
from nequip.utils.versions import check_code_version

default_config = dict(
    root="./",
    tensorboard=False,
    wandb=False,
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "StressForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    device="cuda",
    default_dtype="float32",
    model_dtype="float32",
    allow_tf32=True,
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    gpu_oom_offload=False,
    append=False,
    warn_unused=False,
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
    # Due to what appear to be ongoing bugs with nvFuser, we default to NNC (fuser1) for now:
    # TODO: still default to NNC on CPU regardless even if change this for GPU
    # TODO: default for ROCm?
    _jit_fuser="fuser1",
)


def init_n_update(config):
    import wandb
    from wandb.util import json_friendly_val

    conf_dict = dict(config)
    # wandb mangles keys (in terms of type) as well, but we can't easily correct that because there are many ambiguous edge cases. (E.g. string "-1" vs int -1 as keys, are they different config keys?)
    if any(not isinstance(k, str) for k in conf_dict.keys()):
        raise TypeError(
            "Due to wandb limitations, only string keys are supported in configurations."
        )

    # download from wandb set up
    config.run_id = wandb.util.generate_id()

    wandb.init(
        project=config.wandb_project,
        config=conf_dict,
        group=config.wandb_group,
        name=config.run_name,
        resume="allow",
        id=config.run_id,
    )
    # # download from wandb set up
    updated_parameters = dict(wandb.config)
    for k, v_new in updated_parameters.items():
        skip = False
        if k in config.keys():
            # double check the one sanitized by wandb
            v_old = json_friendly_val(config[k])
            if repr(v_new) == repr(v_old):
                skip = True
        if skip:
            # logging.info(f"# skipping wandb update {k} from {v_old} to {v_new}")
            pass
        else:
            config.update({k: v_new})
            # logging.info(f"# wandb update {k} from {v_old} to {v_new}")
    return config


def main():
    """Entry point for the train bash_app of psiflow

    Should be executed in a temporary directory; this is set as nequip's root
    directory in which the results/wandb folder structure is created. After
    training, only the best model is retained

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to initialized config", default="", type=str
    )
    parser.add_argument(
        "--model", help="path to undeployed model", default="", type=str
    )
    parser.add_argument(
        "--init_only",
        help="only perform initialization",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config, defaults=default_config)

    if args.init_only:  # create dummy validation set based on first state of training
        assert args.model == "None"
        atoms = read(config["dataset_file_name"])
        path_valid = str(Path.cwd() / "validation_dummy.xyz")
        write(path_valid, atoms)
        config["validation_dataset_file_name"] = path_valid

    # put chemical symbols in config
    from ase.data import chemical_symbols
    from ase.io.extxyz import read_extxyz

    with open(config["dataset_file_name"], "r") as f:
        data = list(read_extxyz(f, index=slice(None)))
        ntrain = len(data)
    _all = [set(a.numbers) for a in data]
    numbers = sorted(list(set(b for a in _all for b in a)))
    config["chemical_symbols"] = [chemical_symbols[n] for n in numbers]
    with open(config["validation_dataset_file_name"], "r") as f:
        data = list(read_extxyz(f, index=slice(None)))
        nvalid = len(data)
    config["n_train"] = ntrain
    config["n_val"] = nvalid

    import os

    config["root"] = os.getcwd()
    trainer = fresh_start(config, args.model, args.init_only)
    if not args.init_only:
        trainer.save()
        trainer.train()
    else:
        return 0


def fresh_start(config, path_model, init_only):
    # we use add_to_config cause it's a fresh start and need to record it
    check_code_version(config, add_to_config=False)
    _set_global_options(config)

    # = Make the trainer =
    if config.wandb:
        import wandb  # noqa: F401
        from nequip.train.trainer_wandb import TrainerWandB

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
    if not init_only:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}..."
        )
    else:
        trainer.n_val = 0
        validation_dataset = None

    # = Train/test split =
    print("ntrain: {}".format(trainer.n_train))
    print("nvalid: {}".format(trainer.n_val))
    trainer.set_dataset(dataset, validation_dataset)

    model = model_from_config(config, initialize=True, dataset=trainer.dataset_train)
    logging.info("Successfully built the network...")
    logging.info(
        "psiflow-modification: loading state dict into model; "
        "setting precision float32; transferring to device cuda"
    )
    if not init_only:
        model.load_state_dict(torch.load(path_model, map_location="cpu"))
        model.to(device=torch.device("cuda"), dtype=torch.float32)
    else:
        nequip_config = Config.as_dict(config)
        with open("config.yaml", "w") as f:
            yaml.dump(nequip_config, f, default_flow_style=False)
        torch.save(model.to("cpu").state_dict(), "undeployed.pth")
        return 0
    logging.info("psiflow-modification: success")

    # by doing this here we check also any keys custom builders may have added
    _check_old_keys(config)

    # Equivar test
    # if config.equivariance_test > 0:
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
