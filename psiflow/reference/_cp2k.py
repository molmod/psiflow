from __future__ import annotations  # necessary for type-guarding class methods

import copy
import io
import logging

import numpy as np
import parsl
import typeguard
from ase.data import atomic_numbers
from ase.units import Bohr, Ha
from cp2k_input_tools.generator import CP2KInputGenerator
from cp2k_input_tools.parser import CP2KInputParserSimplified
from parsl.app.app import bash_app, join_app, python_app
from parsl.data_provider.files import File

import psiflow
from psiflow.data import FlowAtoms, NullState
from psiflow.reference.base import BaseReference
from psiflow.utils import copy_app_future

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def check_input(cp2k_input_str: str):
    pass


@typeguard.typechecked
def str_to_dict(cp2k_input_str: str) -> dict:
    return CP2KInputParserSimplified(
        repeated_section_unpack=True,
        # multi_value_unpack=False,
        # level_reduction_blacklist=['KIND'],
    ).parse(io.StringIO(cp2k_input_str))


@typeguard.typechecked
def dict_to_str(cp2k_input_dict: dict) -> str:
    return "\n".join(list(CP2KInputGenerator().line_iter(cp2k_input_dict)))


@typeguard.typechecked
def insert_atoms_in_input(cp2k_input_dict: dict, atoms: FlowAtoms):
    from ase.data import chemical_symbols

    # get rid of topology if it's there
    cp2k_input_dict["force_eval"]["subsys"].pop("topology", None)

    coord = []
    cell = {}
    for i in range(len(atoms)):
        coord.append(
            "{} {} {} {}".format(
                chemical_symbols[atoms.numbers[i]], *atoms.positions[i]
            )
        )
    cp2k_input_dict["force_eval"]["subsys"]["coord"] = {"*": coord}

    assert atoms.pbc.any()  # CP2K needs cell info!
    for i, vector in enumerate(["A", "B", "C"]):
        cell[vector] = "{} {} {}".format(*atoms.cell[i])
    cp2k_input_dict["force_eval"]["subsys"]["cell"] = cell


@typeguard.typechecked
def set_global_section(cp2k_input_dict: dict, properties: tuple):
    global_dict = {}
    if properties == ("energy",):
        global_dict["run_type"] = "ENERGY"
    elif properties == ("energy", "forces"):
        global_dict["run_type"] = "ENERGY_FORCE"
    else:
        raise ValueError("invalid properties: {}".format(properties))

    global_dict["preferred_diag_library"] = "SL"
    global_dict["fm"] = {"type_of_matrix_multiplication": "SCALAPACK"}
    cp2k_input_dict["global"] = global_dict


def parse_cp2k_output(
    cp2k_output_str: str, properties: tuple, atoms: FlowAtoms
) -> FlowAtoms:
    natoms = len(atoms)
    all_lines = cp2k_output_str.split("\n")

    # read coordinates
    lines = None
    for i, line in enumerate(all_lines):
        if line.strip().startswith("MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM"):
            skip = 3
            lines = all_lines[i + skip : i + skip + natoms]
    if lines is None:
        atoms.reference_status = False
        return atoms
    assert len(lines) == natoms
    positions = np.zeros((natoms, 3))
    for j, line in enumerate(lines):
        positions[j, :] = np.array([float(f) for f in line.split()[4:7]])
    assert np.allclose(atoms.get_positions(), positions)

    # try and read energy
    energy = None
    for line in all_lines:
        if line.strip().startswith("ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]"):
            energy = float(line.split()[-1]) * Ha
    if energy is None:
        atoms.reference_status = False
        return atoms
    atoms.reference_status = True
    atoms.info["energy"] = energy
    atoms.arrays.pop("forces", None)  # remove if present for some reason

    # try and read forces if requested
    if "forces" in properties:
        lines = None
        for i, line in enumerate(all_lines):
            if line.strip().startswith("ATOMIC FORCES in [a.u.]"):
                skip = 3
                lines = all_lines[i + skip : i + skip + natoms]
        if lines is None:
            atoms.reference_status = False
            return atoms
        assert len(lines) == natoms
        forces = np.zeros((natoms, 3))
        for j, line in enumerate(lines):
            forces[j, :] = np.array([float(f) for f in line.split()[3:6]])
        forces *= Ha / Bohr
        atoms.arrays["forces"] = forces
    atoms.info.pop("stress", None)  # remove if present for some reason
    return atoms


# typeguarding not compatible with parsl WQEX for some reason
def cp2k_singlepoint_pre(
    atoms: FlowAtoms,
    cp2k_input_dict: dict,
    properties: tuple,
    cp2k_command: str,
    omp_num_threads: int,
    walltime: int = 0,
    stdout: str = "",
    stderr: str = "",
):
    from psiflow.reference._cp2k import (
        dict_to_str,
        insert_atoms_in_input,
        set_global_section,
    )

    set_global_section(cp2k_input_dict, properties)
    insert_atoms_in_input(cp2k_input_dict, atoms)
    if "forces" in properties:
        cp2k_input_dict["force_eval"]["print"] = {"FORCES": {}}
    cp2k_input_str = dict_to_str(cp2k_input_dict)

    # see https://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd = "cd $mytmpdir;"
    command_write = 'echo "{}" > cp2k.inp;'.format(cp2k_input_str)
    command_list = [
        command_tmp,
        command_cd,
        command_write,
        "export OMP_NUM_THREADS={};".format(omp_num_threads),
        "timeout -s 9 {}s".format(walltime - 2),  # kill right before parsl walltime
        cp2k_command,
        "-i cp2k.inp",
        " || true",
    ]
    return " ".join(command_list)


def cp2k_singlepoint_post(
    atoms: FlowAtoms,
    properties: tuple,
    inputs: list[File] = [],
) -> FlowAtoms:
    from psiflow.reference._cp2k import parse_cp2k_output

    atoms.reference_stdout = inputs[0]
    atoms.reference_stderr = inputs[1]
    with open(atoms.reference_stdout, "r") as f:
        cp2k_output_str = f.read()
    return parse_cp2k_output(cp2k_output_str, properties, atoms)


@typeguard.typechecked
class CP2KReference(BaseReference):
    """CP2K Reference

    Arguments
    ---------

    cp2k_input : str
        string representation of the cp2k input file.

    """

    def __init__(self, cp2k_input_str: str, **kwargs):
        check_input(cp2k_input_str)
        self.cp2k_input_str = cp2k_input_str
        self.cp2k_input_dict = str_to_dict(cp2k_input_str)
        super().__init__(**kwargs)

    def get_single_atom_references(self, element):
        number = atomic_numbers[element]
        references = []
        for mult in range(1, 16):
            if number % 2 == 0 and mult % 2 == 0:
                continue  # not 2N + 1 is never even
            if mult - 1 > number:
                continue  # max S = 2 * (N * 1/2) + 1
            cp2k_input_dict = copy.deepcopy(self.cp2k_input_dict)
            cp2k_input_dict["force_eval"]["dft"]["uks"] = "TRUE"
            cp2k_input_dict["force_eval"]["dft"]["multiplicity"] = mult
            cp2k_input_dict["force_eval"]["dft"]["charge"] = 0
            cp2k_input_dict["force_eval"]["dft"]["xc"].pop("vdw_potential", None)
            if "scf" in cp2k_input_dict["force_eval"]["dft"]:
                if "ot" in cp2k_input_dict["force_eval"]["dft"]["scf"]:
                    cp2k_input_dict["force_eval"]["dft"]["scf"]["ot"][
                        "minimizer"
                    ] = "CG"
                else:
                    cp2k_input_dict["force_eval"]["dft"]["scf"]["ot"] = {
                        "minimizer": "CG"
                    }
            else:
                cp2k_input_dict["force_eval"]["dft"]["scf"] = {
                    "ot": {"minimizer": "CG"}
                }

            reference = CP2KReference(dict_to_str(cp2k_input_dict))
            references.append((mult, reference))
        return references

    @property
    def parameters(self):
        return {
            "cp2k_input_dict": copy.deepcopy(self.cp2k_input_dict),
            "properties": self.properties,
        }

    @classmethod
    def create_apps(cls):
        context = psiflow.context()
        definition = context[cls]
        label = definition.name()
        mpi_command = definition.mpi_command
        ncores = definition.cores_per_worker
        walltime = definition.max_walltime

        # parse full command
        omp_num_threads = 1
        command = ""
        command += mpi_command(ncores)
        command += " "
        command += "cp2k.psmp"

        singlepoint_pre = bash_app(
            cp2k_singlepoint_pre,
            executors=[label],
        )
        singlepoint_post = python_app(
            cp2k_singlepoint_post,
            executors=["default_threads"],
        )

        @join_app
        def singlepoint_wrapped(
            atoms,
            parameters,
        ):
            if atoms == NullState:
                return copy_app_future(NullState)
            else:
                pre = singlepoint_pre(
                    atoms,
                    parameters["cp2k_input_dict"],
                    parameters["properties"],
                    command,
                    omp_num_threads,
                    stdout=parsl.AUTO_LOGNAME,
                    stderr=parsl.AUTO_LOGNAME,
                    walltime=60 * walltime,  # killed after walltime - 2s
                )
                return singlepoint_post(
                    atoms=atoms,
                    properties=parameters["properties"],
                    inputs=[pre.stdout, pre.stderr, pre],  # wait for bash app
                )

        context.register_app(cls, "evaluate_single", singlepoint_wrapped)
        super(CP2KReference, cls).create_apps()
