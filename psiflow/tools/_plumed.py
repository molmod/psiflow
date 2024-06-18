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

# from .function import Function


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


# @typeguard.typechecked
# @psiflow.serializable
# class PlumedFunction(Function):
#     outputs = ('energy', 'forces', 'stress')
#     _plumed_input: str
#     external: Optional[psiflow._DataFuture]
# 
#     def __init__(
#         self,
#         plumed_input: str,
#         external: Union[str, Path, File, None] = None,
#     ):
#         _plumed_input = remove_comments_printflush(plumed_input)
# 
#         if type(external) in [str, Path]:
#             external = File(str(external))
#         if external is not None:
#             assert external.filepath in _plumed_input
#             _plumed_input = _plumed_input.replace(external.filepath, "PLACEHOLDER")
#         self._plumed_input = _plumed_input
#         self.external = external
#         self._create_apps()
# 
#     def create_apps(self):
#         pass
# 
#     def plumed_input(self):
#         plumed_input = self._plumed_input
#         if self.external is not None:
#             plumed_input = plumed_input.replace("PLACEHOLDER", self.external.filepath)
#         return plumed_input
# 
#     def parameters(self):
#         return {
#             "plumed_input": self.plumed_input(),
#         }
# 
#     @staticmethod
#     def apply(
#         geometries: list[Geometry],
#         plumed_input: str,
#         external: Optional[File] = None,
#         insert: bool = False,
#         outputs: Optional[list[str]] = None,
#     ) -> list[np.ndarray]:
#         energy, forces, stress = create_outputs(
#             EinsteinCrystalFunction.outputs,
#             geometries,
#         )
# 
#         def geometry_to_key(geometry: Geometry) -> tuple:
#             return tuple([geometry.periodic]) + tuple(geometry.per_atom.numbers)
# 
#         plumed_instances = {}
#         for i, geometry in enumerate(geometries):
#             if geometry == NullState:
#                 continue
#             key = geometry_to_key(geometry)
#             if key not in plumed_instances:
#                 from plumed import Plumed
#                 tmp = tempfile.NamedTemporaryFile(
#                     prefix="plumed_", mode="w+", delete=False
#                 )
#                 # plumed creates a back up if this file would already exist
#                 os.remove(tmp.name)
#                 plumed_ = Plumed()
#                 ps = 1000 * fs
#                 plumed_.cmd("setRealPrecision", 8)
#                 plumed_.cmd("setMDEnergyUnits", mol / kJ)
#                 plumed_.cmd("setMDLengthUnits", 1 / nm)
#                 plumed_.cmd("setMDTimeUnits", 1 / ps)
#                 plumed_.cmd("setMDChargeUnits", 1.0)
#                 plumed_.cmd("setMDMassUnits", 1.0)
# 
#                 plumed_.cmd("setLogFile", tmp.name)
#                 plumed_.cmd("setRestart", True)
#                 plumed_.cmd("setNatoms", len(geometry))
#                 plumed_.cmd("init")
#                 plumed_input = self.plumed_input
#                 for line in plumed_input.split("\n"):
#                     plumed_.cmd("readInputLine", line)
#                 os.remove(tmp.name)  # remove whatever plumed has created
#                 plumed_instances[key] = plumed_
# 
#             plumed_ = plumed_instances[key]
#             if geometry.periodic:
#                 cell = np.copy(geometry.cell).astype(np.float64)
#                 plumed_.cmd("setBox", cell)
# 
#             # set positions
#             energy = np.zeros((1,))
#             forces = np.zeros((len(geometry), 3))
#             virial = np.zeros((3, 3))
#             masses = np.array([atomic_masses[n] for n in geometry.per_atom.numbers])
#             plumed_.cmd("setStep", 0)
#             plumed_.cmd("setPositions", geometry.per_atom.positions.astype(np.float64))
#             plumed_.cmd("setMasses", masses)
#             plumed_.cmd("setForces", forces)
#             plumed_.cmd("setVirial", virial)
#             plumed_.cmd("prepareCalc")
#             plumed_.cmd("performCalcNoUpdate")
#             plumed_.cmd("getBias", energy)
#             if geometry.periodic:
#                 stress = virial / np.linalg.det(geometry.cell)
#             else:
#                 stress = np.zeros((3, 3))
# 
#             outputs['energy'][i] = float(energy.item())
#             outputs['forces'][i, :len(geometry)] = forces
#             outputs['stress'][i] = stress
#         return outputs
