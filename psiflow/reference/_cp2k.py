from __future__ import annotations  # necessary for type-guarding class methods

import copy
import io
import logging
from functools import partial
from typing import Optional, Union

import numpy as np
import typeguard
from ase.data import atomic_numbers
from ase.units import Bohr, Ha
from cp2k_input_tools.generator import CP2KInputGenerator
from cp2k_input_tools.parser import CP2KInputParserSimplified
from parsl.app.app import bash_app, python_app

import psiflow
from psiflow.geometry import Geometry, NullState
from psiflow.reference.reference import Reference
from psiflow.utils import TMP_COMMAND, CD_COMMAND

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
def insert_atoms_in_input(cp2k_input_dict: dict, geometry: Geometry):
    from ase.data import chemical_symbols

    # get rid of topology if it's there
    cp2k_input_dict["force_eval"]["subsys"].pop("topology", None)

    coord = []
    cell = {}
    numbers = geometry.per_atom.numbers
    positions = geometry.per_atom.positions
    for i in range(len(geometry)):
        coord.append("{} {} {} {}".format(chemical_symbols[numbers[i]], *positions[i]))
    cp2k_input_dict["force_eval"]["subsys"]["coord"] = {"*": coord}

    assert geometry.periodic  # CP2K needs cell info!
    for i, vector in enumerate(["A", "B", "C"]):
        cell[vector] = "{} {} {}".format(*geometry.cell[i])
    cp2k_input_dict["force_eval"]["subsys"]["cell"] = cell


@typeguard.typechecked
def set_global_section(cp2k_input_dict: dict, properties: tuple):
    if "global" not in cp2k_input_dict:
        cp2k_input_dict["global"] = {}
    global_dict = cp2k_input_dict["global"]

    # override low/silent print levels
    level = global_dict.pop("print_level", "MEDIUM")
    if level in ["SILENT", "LOW"]:
        global_dict["print_level"] = "MEDIUM"

    if properties == ("energy",):
        global_dict["run_type"] = "ENERGY"
    elif properties == ("energy", "forces"):
        global_dict["run_type"] = "ENERGY_FORCE"
    else:
        raise ValueError("invalid properties: {}".format(properties))

    if "preferred_diag_library" not in global_dict:
        global_dict["preferred_diag_library"] = "SL"
    if "fm" not in global_dict:
        global_dict["fm"] = {"type_of_matrix_multiplication": "SCALAPACK"}


def parse_cp2k_output(
    cp2k_output_str: str, properties: tuple, geometry: Geometry
) -> Geometry:
    natoms = len(geometry)
    all_lines = cp2k_output_str.split("\n")

    # read coordinates
    lines = None
    for i, line in enumerate(all_lines):
        if line.strip().startswith("MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM"):
            skip = 3
            lines = all_lines[i + skip : i + skip + natoms]
    if lines is None:
        return NullState
    assert len(lines) == natoms
    positions = np.zeros((natoms, 3))
    for j, line in enumerate(lines):
        try:
            positions[j, :] = np.array([float(f) for f in line.split()[4:7]])
        except ValueError:  # if positions exploded, CP2K puts *** instead of float
            return NullState
    assert np.allclose(
        geometry.per_atom.positions, positions, atol=1e-2
    )  # accurate up to 0.01 A

    # try and read energy
    energy = None
    for line in all_lines:
        if line.strip().startswith("ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]"):
            energy = float(line.split()[-1]) * Ha
    if energy is None:
        return NullState
    geometry.energy = energy
    geometry.per_atom.forces[:] = np.nan

    # try and read forces if requested
    if "forces" in properties:
        lines = None
        for i, line in enumerate(all_lines):
            if line.strip().startswith("ATOMIC FORCES in [a.u.]"):
                skip = 3
                lines = all_lines[i + skip : i + skip + natoms]
        if lines is None:
            return NullState
        assert len(lines) == natoms
        forces = np.zeros((natoms, 3))
        for j, line in enumerate(lines):
            forces[j, :] = np.array([float(f) for f in line.split()[3:6]])
        forces *= Ha / Bohr
        geometry.per_atom.forces[:] = forces
    geometry.stress = None
    return geometry


def _prepare_input(
    geometry: Geometry,
    cp2k_input_dict: dict = {},
    properties: tuple = (),
    outputs: list = [],
):
    from psiflow.reference._cp2k import (
        dict_to_str,
        insert_atoms_in_input,
        set_global_section,
    )

    set_global_section(cp2k_input_dict, properties)
    insert_atoms_in_input(cp2k_input_dict, geometry)
    if "forces" in properties:
        cp2k_input_dict["force_eval"]["print"] = {"FORCES": {}}
    cp2k_input_str = dict_to_str(cp2k_input_dict)
    with open(outputs[0], "w") as f:
        f.write(cp2k_input_str)


prepare_input = python_app(_prepare_input, executors=["default_threads"])


# typeguarding for some reason incompatible with WQ
def cp2k_singlepoint_pre(
    cp2k_command: str = "",
    stdout: str = "",
    stderr: str = "",
    inputs: list = [],
    parsl_resource_specification: Optional[dict] = None,
):
    cp_command = f"cp {inputs[0].filepath} cp2k.inp"
    command_list = [TMP_COMMAND, CD_COMMAND, cp_command, cp2k_command]
    return " && ".join(command_list)


@typeguard.typechecked
def cp2k_singlepoint_post(
    geometry: Geometry,
    properties: tuple = (),
    inputs: list = [],
) -> Geometry:
    from psiflow.geometry import NullState, new_nullstate
    from psiflow.reference._cp2k import parse_cp2k_output

    with open(inputs[0], "r") as f:
        cp2k_output_str = f.read()
    geometry = parse_cp2k_output(cp2k_output_str, properties, geometry)
    if geometry != NullState:
        geometry.stdout = inputs[0]
    else:  # a little hacky
        geometry = new_nullstate()
        geometry.stdout = inputs[0]
    return geometry


@typeguard.typechecked
@psiflow.serializable
class CP2K(Reference):
    outputs: list
    executor: str
    cp2k_input_str: str
    cp2k_input_dict: dict

    def __init__(
        self,
        cp2k_input_str: str,
        executor: str = "CP2K",
        outputs: Union[tuple, list] = ("energy", "forces"),
    ):
        self.executor = executor
        check_input(cp2k_input_str)
        self.cp2k_input_str = cp2k_input_str
        self.cp2k_input_dict = str_to_dict(cp2k_input_str)
        self.outputs = list(outputs)
        self._create_apps()

    def _create_apps(self):
        definition = psiflow.context().definitions[self.executor]
        cp2k_command = definition.command()
        wq_resources = definition.wq_resources()
        app_pre = bash_app(cp2k_singlepoint_pre, executors=[self.executor])
        app_post = python_app(cp2k_singlepoint_post, executors=["default_threads"])

        # create wrapped pre app which first parses the input file and writes it to
        # disk, then call the actual bash app with the input file as a DataFuture dependency
        # This is necessary because for very large structures, the size of the cp2k input
        # file is too long to pass as an argument in a command line
        def wrapped_app_pre(geometry, stdout: str, stderr: str):
            future = prepare_input(
                geometry,
                cp2k_input_dict=copy.deepcopy(self.cp2k_input_dict),
                properties=tuple(self.outputs),
                outputs=[psiflow.context().new_file("cp2k_", ".inp")],
            )
            return app_pre(
                cp2k_command=cp2k_command,
                stdout=stdout,
                stderr=stderr,
                inputs=[future.outputs[0]],
                parsl_resource_specification=wq_resources,
            )

        self.app_pre = wrapped_app_pre
        self.app_post = partial(
            app_post,
            properties=tuple(self.outputs),
        )

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
            # necessary for oxygen calculation, at least in 2024.1
            key = "ignore_convergence_failure"
            cp2k_input_dict["force_eval"]["dft"]["scf"][key] = "TRUE"

            reference = CP2K(
                dict_to_str(cp2k_input_dict),
                outputs=self.outputs,
                executor=self.executor,
            )
            references.append((mult, reference))
        return references
