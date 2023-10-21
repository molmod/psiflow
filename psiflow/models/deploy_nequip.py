from typing import Final, Optional
import argparse
import logging
import yaml
import packaging.version

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

import torch

from e3nn.util.jit import script

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.utils.versions import check_code_version, get_current_code_versions
from nequip.scripts.train import default_config
from nequip.utils.misc import dtype_to_name
from nequip.utils._global_options import _set_global_options

CONFIG_KEY: Final[str] = "config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
TORCH_VERSION_KEY: Final[str] = "torch_version"
E3NN_VERSION_KEY: Final[str] = "e3nn_version"
CODE_COMMITS_KEY: Final[str] = "code_commits"
R_MAX_KEY: Final[str] = "r_max"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
JIT_FUSION_STRATEGY: Final[str] = "_jit_fusion_strategy"
TF32_KEY: Final[str] = "allow_tf32"
DEFAULT_DTYPE_KEY: Final[str] = "default_dtype"
MODEL_DTYPE_KEY: Final[str] = "model_dtype"

_ALL_METADATA_KEYS = [
    CONFIG_KEY,
    NEQUIP_VERSION_KEY,
    TORCH_VERSION_KEY,
    E3NN_VERSION_KEY,
    R_MAX_KEY,
    N_SPECIES_KEY,
    TYPE_NAMES_KEY,
    JIT_BAILOUT_KEY,
    JIT_FUSION_STRATEGY,
    TF32_KEY,
    DEFAULT_DTYPE_KEY,
    MODEL_DTYPE_KEY,
]


def _register_metadata_key(key: str) -> None:
    _ALL_METADATA_KEYS.append(key)


_current_metadata: Optional[dict] = None


def _set_deploy_metadata(key: str, value) -> None:
    # TODO: not thread safe but who cares?
    global _current_metadata
    if _current_metadata is None:
        pass  # not deploying right now
    elif key in _current_metadata:
        raise RuntimeError(f"{key} already set in the deployment metadata")
    else:
        _current_metadata[key] = value


def _compile_for_deploy(model):
    model.eval()

    if not isinstance(model, torch.jit.ScriptModule):
        model = script(model)

    return model


def main():
    """Entry point for deploy bash app"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to initialized config", default="", type=str
    )
    parser.add_argument(
        "--model", help="path to undeployed model", default="", type=str
    )
    parser.add_argument(
        "--deployed", help="path to deployed model", default="", type=str
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, "INFO"))
    state_dict = torch.load(str(args.model), map_location="cpu")
    config = Config.from_file(str(args.config), defaults=default_config)

    _set_global_options(config)
    check_code_version(config)

    # build the actual model]
    # reset the global metadata dict so that model builders can fill it:
    global _current_metadata
    _current_metadata = {}

    model = model_from_config(config, initialize=False)
    model.load_state_dict(state_dict, strict=True)

    # -- compile --
    model = _compile_for_deploy(model)
    logging.info("Compiled & optimized model.")

    # Deploy
    metadata: dict = {}
    code_versions, code_commits = get_current_code_versions(config)
    for code, version in code_versions.items():
        metadata[code + "_version"] = version
    if len(code_commits) > 0:
        metadata[CODE_COMMITS_KEY] = ";".join(
            f"{k}={v}" for k, v in code_commits.items()
        )

    metadata[R_MAX_KEY] = str(float(config["r_max"]))
    n_species = str(config["num_types"])
    type_names = config["type_names"]
    metadata[N_SPECIES_KEY] = str(n_species)
    metadata[TYPE_NAMES_KEY] = " ".join(type_names)

    metadata[JIT_BAILOUT_KEY] = str(config[JIT_BAILOUT_KEY])
    if (
        packaging.version.parse(torch.__version__) >= packaging.version.parse("1.11")
        and JIT_FUSION_STRATEGY in config
    ):
        metadata[JIT_FUSION_STRATEGY] = ";".join(
            "%s,%i" % e for e in config[JIT_FUSION_STRATEGY]
        )
    metadata[TF32_KEY] = str(int(config["allow_tf32"]))
    metadata[DEFAULT_DTYPE_KEY] = dtype_to_name(config["default_dtype"])
    metadata[MODEL_DTYPE_KEY] = dtype_to_name(config["model_dtype"])
    metadata[CONFIG_KEY] = yaml.dump(Config.as_dict(config))

    for k, v in _current_metadata.items():
        if k in metadata:
            raise RuntimeError(f"Custom deploy key {k} was already set")
        metadata[k] = v
    _current_metadata = None

    metadata = {k: v.encode("ascii") for k, v in metadata.items()}

    torch.jit.save(model, args.deployed, _extra_files=metadata)
