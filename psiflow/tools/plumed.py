import logging
import tempfile
import os
from typing import Optional, Union
from pathlib import Path

import numpy as np
from ase.units import kJ, mol, nm, fs
from ase.data import atomic_masses
import typeguard
from parsl.data_provider.files import File

import psiflow
from psiflow.geometry import Geometry, NullState


@typeguard.typechecked
def try_manual_plumed_linking() -> str:
    if "PLUMED_KERNEL" not in os.environ.keys():
        # try linking manually
        if "CONDA_PREFIX" in os.environ.keys():  # for conda environments
            p = "CONDA_PREFIX"
        elif "PREFIX" in os.environ.keys():  # for pip environments
            p = "PREFIX"
        else:
            raise ValueError("failed to set plumed .so kernel")
        path = os.environ[p] + "/lib/libplumedKernel.so"
        if os.path.exists(path):
            os.environ["PLUMED_KERNEL"] = path
            logging.info("plumed kernel manually set at : {}".format(path))
        else:
            raise ValueError("plumed kernel not found at {}".format(path))
    return os.environ["PLUMED_KERNEL"]


@typeguard.typechecked
def remove_comments_printflush(plumed_input: str) -> str:
    new_input = []
    for line in list(plumed_input.split("\n")):
        pre_comment = line.strip().split("#")[0].strip()
        if len(pre_comment) == 0:
            continue
        if pre_comment.startswith("PRINT"):
            continue
        if pre_comment.startswith("FLUSH"):
            continue
        new_input.append(pre_comment)
    return "\n".join(new_input)


@typeguard.typechecked
def set_path_in_plumed(plumed_input: str, keyword: str, path_to_set: str) -> str:
    lines = plumed_input.split("\n")
    for i, line in enumerate(lines):
        if keyword in line:
            if "FILE=" not in line:
                lines[i] = line + " FILE={}".format(path_to_set)
                continue
            line_before = line.split("FILE=")[0]
            line_after = line.split("FILE=")[1].split()[1:]
            lines[i] = (
                line_before + "FILE={} ".format(path_to_set) + " ".join(line_after)
            )
    return "\n".join(lines)
