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
from parsl.app.app import bash_app, join_app, python_app
from parsl.app.bash import BashApp
from parsl.app.python import PythonApp
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Geometry
from psiflow.reference.reference import Reference

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
def insert_atoms_in_input(cp2k_input_dict: dict, atoms: Geometry):
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
    cp2k_output_str: str, properties: tuple, atoms: Geometry
) -> Geometry:
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
    assert np.allclose(
        atoms.get_positions(), positions, atol=1e-2
    )  # accurate up to 0.01 A

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


# typeguarding for some reason incompatible with WQ
def cp2k_singlepoint_pre(
    atoms: Geometry,
    cp2k_input_dict: dict,
    properties: tuple,
    cp2k_command: str,
    stdout: str = "",
    stderr: str = "",
    parsl_resource_specification: Optional[dict] = None,
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
    tmp_command = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    cd_command = "cd $mytmpdir;"
    write_command = 'echo "{}" > cp2k.inp;'.format(cp2k_input_str)
    command_list = [
        tmp_command,
        cd_command,
        write_command,
        cp2k_command,
    ]
    return " ".join(command_list)


def cp2k_singlepoint_post(
    atoms: Geometry,
    properties: tuple,
    inputs: list = [],
) -> Geometry:
    from psiflow.data import NullState
    from psiflow.reference._cp2k import parse_cp2k_output

    if atoms == NullState:
        return NullState.copy()

    atoms.reference_stdout = inputs[0]
    atoms.reference_stderr = inputs[1]
    with open(atoms.reference_stdout, "r") as f:
        cp2k_output_str = f.read()
    return parse_cp2k_output(cp2k_output_str, properties, atoms)


@typeguard.typechecked
@join_app
def evaluate_single(
    atoms: Union[Geometry, AppFuture],
    cp2k_input_dict: dict,
    properties: tuple,
    cp2k_command: str,
    wq_resources: dict[str, Union[float, int]],
    app_pre: BashApp,
    app_post: PythonApp,
) -> AppFuture:
    import parsl

    from psiflow.data import NullState
    from psiflow.utils import copy_app_future

    if atoms == NullState:
        return copy_app_future(NullState)
    else:
        pre = app_pre(
            atoms,
            cp2k_input_dict,
            properties,
            cp2k_command=cp2k_command,
            stdout=parsl.AUTO_LOGNAME,
            stderr=parsl.AUTO_LOGNAME,
            parsl_resource_specification=wq_resources,
        )
        return app_post(
            atoms=atoms,
            properties=properties,
            inputs=[pre.stdout, pre.stderr, pre],  # wait for bash app
        )


@typeguard.typechecked
@psiflow.serializable
class CP2K(Reference):
    cp2k_input_str: str
    cp2k_input_dict: dict

    def __init__(self, cp2k_input_str: str, **kwargs):
        super().__init__(**kwargs)
        check_input(cp2k_input_str)
        self.cp2k_input_str = cp2k_input_str
        self.cp2k_input_dict = str_to_dict(cp2k_input_str)
        self._create_apps()

    def _create_apps(self):
        definition = psiflow.context().definitions["ReferenceEvaluation"]
        cp2k_command = definition.cp2k_command()
        executor_label = definition.name
        wq_resources = definition.wq_resources()
        app_pre = bash_app(cp2k_singlepoint_pre, executors=[executor_label])
        app_post = python_app(cp2k_singlepoint_post, executors=["default_threads"])
        self.evaluate_single = partial(
            evaluate_single,
            cp2k_input_dict=self.cp2k_input_dict,
            properties=self.properties,
            cp2k_command=cp2k_command,
            wq_resources=wq_resources,
            app_pre=app_pre,
            app_post=app_post,
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

            reference = CP2K(dict_to_str(cp2k_input_dict))
            references.append((mult, reference))
        return references
