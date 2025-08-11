from __future__ import annotations  # necessary for type-guarding class methods

import copy
import io
import warnings
from typing import Optional, Union

import numpy as np
from ase.data import chemical_symbols
from ase.units import Bohr, Ha
from cp2k_input_tools.generator import CP2KInputGenerator
from cp2k_input_tools.parser import CP2KInputParserSimplified
from parsl import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference
from psiflow.reference.utils import (
    find_line,
    get_spin_multiplicities,
    lines_to_array,
    Status,
    copy_data_to_geometry,
)
from psiflow.utils import TMP_COMMAND, CD_COMMAND


# costly to initialise
input_parser = CP2KInputParserSimplified(
    repeated_section_unpack=True,
    # multi_value_unpack=False,
    # level_reduction_blacklist=['KIND'],
)
input_generator = CP2KInputGenerator()


def str_to_dict(input_str: str) -> dict:
    return input_parser.parse(io.StringIO(input_str))


def dict_to_str(input_dict: dict) -> str:
    return "\n".join(list(input_generator.line_iter(input_dict)))


def modify_input(input_dict: dict, properties: tuple) -> None:
    global_dict = input_dict.setdefault("global", {})

    # override low/silent print levels
    if global_dict.get("print_level") in ["SILENT", "LOW"]:
        global_dict["print_level"] = "MEDIUM"

    if properties == ("energy",):
        global_dict["run_type"] = "ENERGY"
    elif properties == ("energy", "forces"):
        # output forces
        global_dict["run_type"] = "ENERGY_FORCE"
        input_dict["force_eval"]["print"] = {"FORCES": {}}
    else:
        raise ValueError("invalid properties: {}".format(properties))

    if "preferred_diag_library" not in global_dict:
        global_dict["preferred_diag_library"] = "SL"
    if "fm" not in global_dict:
        global_dict["fm"] = {"type_of_matrix_multiplication": "SCALAPACK"}


def parse_output(output_str: str, properties: tuple) -> dict[str, float | np.ndarray]:
    lines = output_str.split("\n")
    data = {}

    # output status
    idx = find_line(lines, "CP2K", reverse=True, max_lines=250)
    data["status"] = status = Status.SUCCESS if idx is not None else Status.FAILED
    if status == Status.SUCCESS:
        # total runtime
        data["runtime"] = float(lines[idx].split()[-1])

    # find number of atoms
    idx = find_line(lines, "TOTAL NUMBERS AND MAXIMUM NUMBERS")
    idx = find_line(lines, "- Atoms:", idx)
    natoms = data["natoms"] = int(lines[idx].split()[-1])

    # read coordinates
    key = "MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM"
    idx = find_line(lines, key, idx) + 3
    data["positions"] = lines_to_array(lines[idx : idx + natoms], 4, 7)

    # read energy
    key = "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]"
    idx = find_line(lines, key, idx)
    data["energy"] = float(lines[idx].split()[-1]) * Ha

    if "forces" not in properties:
        return data

    # read forces
    key = "ATOMIC FORCES in [a.u.]"
    idx = find_line(lines, key, idx) + 3
    forces = lines_to_array(lines[idx : idx + natoms], 3)

    return data | {"forces": forces * Ha / Bohr}


@psiflow.serializable
class CP2K(Reference):
    _execute_label = "cp2k_singlepoint"

    def __init__(
        self,
        input_str: str,
        executor: str = "CP2K",
        outputs: Union[tuple, list] = ("energy", "forces"),
    ):
        self.executor = executor
        self.outputs = tuple(outputs)
        self.input_dict = str_to_dict(input_str)
        modify_input(self.input_dict, outputs)
        self._create_apps()

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture[float]:
        assert box_size, "CP2K expects a periodic box."
        return super().compute_atomic_energy(element, box_size)

    def get_single_atom_references(self, element: str) -> dict[int, Reference]:
        input_dict = copy.deepcopy(self.input_dict)
        input_section = input_dict["force_eval"]["dft"]
        input_section |= {"uks": "TRUE", "charge": 0, "multiplicity": "{mult}"}
        input_section["xc"].pop("vdw_potential", None)  # no dispersion
        if "scf" in input_section:
            if "ot" in input_section["scf"]:
                input_section["scf"]["ot"]["minimizer"] = "CG"
            else:
                input_section["scf"]["ot"] = {"minimizer": "CG"}
        else:
            input_section["scf"] = {"ot": {"minimizer": "CG"}}
        # necessary for oxygen calculation, at least in 2024.1
        input_section["scf"]["ignore_convergence_failure"] = "TRUE"
        input_str = dict_to_str(input_dict)

        references = {}
        for mult in get_spin_multiplicities(element):
            references[mult] = CP2K(
                input_str.format(mult=mult),
                outputs=self.outputs,
                executor=self.executor,
            )
        return references

    def get_shell_command(self, inputs: list[File]) -> str:
        command_list = [
            TMP_COMMAND,
            CD_COMMAND,
            f"cp {inputs[0].filepath} cp2k.inp",
            self.execute_command,
        ]
        return "\n".join(command_list)

    def parse_output(self, stdout: str) -> dict:
        return parse_output(stdout, self.outputs)

    def create_input(self, geom: Geometry) -> tuple[bool, Optional[File]]:
        if not geom.periodic:
            msg = "CP2K expects periodic boundary conditions, skipping geometry"
            warnings.warn(msg)
            return False, None

        section = self.input_dict["force_eval"]["subsys"]
        section_copy = copy.deepcopy(section)
        section.pop("topology", None)  # remove topology section

        # insert geometry and cell
        symbols = np.array(chemical_symbols)[geom.per_atom.numbers]
        coord = [
            f"{s:5} {p[0]:<15.8f} {p[1]:<15.8f} {p[2]:<15.8f}"
            for s, p in zip(symbols, geom.per_atom.positions)
        ]
        section["coord"] = {"*": coord}
        cell = {}
        for i, vector in enumerate(["A", "B", "C"]):
            cell[vector] = "{} {} {}".format(*geom.cell[i])
        section["cell"] = cell

        with open(file := psiflow.context().new_file("cp2k_", ".inp"), "w") as f:
            f.write(dict_to_str(self.input_dict))

        self.input_dict["force_eval"]["subsys"] = section_copy  # revert changes
        return True, File(file)
