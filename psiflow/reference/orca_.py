from __future__ import annotations  # necessary for type-guarding class methods

import warnings
import re
from functools import partial
from typing import Optional, Union

import ase.symbols
import numpy as np
from ase.units import Bohr, Ha
from parsl import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference, Status, get_spin_multiplicities
from psiflow.utils import TMP_COMMAND, CD_COMMAND
from psiflow.utils.parse import find_line, lines_to_array, string_to_timedelta


KEY_GHOST = "ghost"
KEY_DUMMY = 0
SYMBOLS = np.array(ase.symbols.chemical_symbols)
SYMBOLS[KEY_DUMMY] = "DA"
DEFAULT_KWARGS = dict(charge=0, multiplicity=1)


def format_block(key: str, keywords: list[str]) -> str:
    """"""
    return "\n".join([f"%{key}", *[f"\t{s}" for s in keywords], "end"])


def create_input_template(
    task: str = "EnGrad",
    method: str = "HF",
    basis: str = "def2-SVP",
    comment: str = "",
    keywords: list[str] = (),
    blocks: dict[str, list[str]] = None,
) -> str:
    """"""
    blocks = blocks or {}
    lines = [
        f"# {comment}",
        f'! {task} {method} {basis} {" ".join(keywords)}',
        *[format_block(k, v) for k, v in blocks.items()],
    ]
    return "\n".join(lines)


def format_coord(geom: Geometry) -> str:
    """"""
    # TODO: ghost atoms?
    symbols = SYMBOLS[geom.per_atom.numbers]
    # if KEY_GHOST in atoms.arrays:
    #     for idx in np.flatnonzero(atoms.arrays[KEY_GHOST]):
    #         symbols[idx] = f"{symbols[idx]}:"
    data = [
        f"{s:5} {p[0]:<15.8f} {p[1]:<15.8f} {p[2]:<15.8f}"
        for s, p in zip(symbols, geom.per_atom.positions)
    ]
    return "\n".join(data)


def check_input(input_template: str, properties: tuple[str, ...]) -> str:
    """"""
    research = partial(re.search, string=input_template, flags=re.IGNORECASE)

    compute_forces = bool(research("Engrad") or research("Numgrad"))
    if compute_forces and "forces" not in properties:
        msg = "ORCA input asks for gradients, but 'forces' not in outputs."
        warnings.warn(msg)
    elif not compute_forces and "forces" in properties:
        msg = "ORCA input does not ask for gradients, but 'forces' in outputs."
        warnings.warn(msg)

    # add some placeholder blocks
    lines = [input_template]
    if not research("%maxcore"):
        lines.append("%maxcore {memory}")
    if not research("%pal"):
        lines.append(format_block("pal", ["nprocs {cores}"]))
    if not research("\*xyz"):
        lines.append("*xyz {charge} {multiplicity}\n{coord}\n*")

    return "\n".join(lines)


def parse_output(stdout: str, properties: tuple[str, ...]) -> dict:
    lines = stdout.split("\n")
    data = {}

    # output status
    line = "****ORCA TERMINATED NORMALLY****"
    idx = find_line(lines, line, reverse=True, max_lines=5)
    data["status"] = status = Status.SUCCESS if idx is not None else Status.FAILED
    if status == Status.SUCCESS:
        # total runtime
        idx = find_line(lines, "TOTAL RUN TIME", reverse=True, max_lines=5)
        data["runtime"] = string_to_timedelta(lines[idx][16:])

    # read coordinates
    idx_start = idx = find_line(lines, "CARTESIAN COORDINATES (ANGSTROEM)") + 2
    idx_stop = idx = find_line(lines, "---", idx) - 1
    positions = lines_to_array(lines[idx_start:idx_stop], start=1, stop=4)
    data["positions"], data["natoms"] = positions, positions.shape[0]

    # read energy
    # TODO: not exactly equal to log file - different conversion factor?
    idx = find_line(lines, "FINAL SINGLE POINT ENERGY", idx)
    data["energy"] = float(lines[idx].split()[-1]) * Ha

    if "forces" not in properties:
        return data

    # read forces
    idx_start = idx = find_line(lines, "CARTESIAN GRADIENT", idx) + 3
    idx_stop = find_line(lines, "Difference", idx) - 1
    gradients = lines_to_array(lines[idx_start:idx_stop], start=3, stop=6)
    data["forces"] = - gradients * Ha / Bohr

    return data


@psiflow.serializable
class ORCA(Reference):
    _execute_label = "orca_singlepoint"
    input_template: str
    input_kwargs: dict

    def __init__(
        self,
        input_template: str,
        executor: str = "ORCA",
        outputs: Union[tuple, list] = ("energy", "forces"),
    ):
        self.executor = executor
        self.input_template = check_input(input_template, outputs)
        self.input_kwargs = DEFAULT_KWARGS.copy()  # TODO: user control?
        self.outputs = tuple(outputs)
        self._create_apps()

    def _create_apps(self):
        super()._create_apps()
        definition = psiflow.context().definitions[self.executor]
        wq_resources = definition.wq_resources()
        cores, memory = wq_resources["cores"], wq_resources["memory"]
        self.input_kwargs |= {"cores": cores, "memory": memory // cores}  # in MB

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture[float]:
        assert box_size is None, "ORCA does not do PBC"
        return super().compute_atomic_energy(element)

    def get_single_atom_references(self, element: str) -> dict[int, Reference]:
        # TODO: this is not properly tested - special options?
        # ORCA automatically switches to UHF/UKS
        references = {}
        for mult in get_spin_multiplicities(element):
            ref = ORCA(
                self.input_template, outputs=self.outputs, executor=self.executor
            )
            ref.input_kwargs["multiplicity"] = mult
            references[mult] = ref
        return references

    def get_shell_command(self, inputs: list[File]) -> str:
        command_list = [
            TMP_COMMAND,
            CD_COMMAND,
            f"cp {inputs[0].filepath} orca.inp",
            self.execute_command,
        ]
        return "\n".join(command_list)

    def parse_output(self, stdout: str) -> dict:
        return parse_output(stdout, self.outputs)

    def create_input(self, geom: Geometry) -> tuple[bool, Optional[File]]:
        if geom.periodic:
            msg = "ORCA does not support periodic boundary conditions"
            warnings.warn(msg)
            return False, None

        input_str = self.input_template.format(
            coord=format_coord(geom), **self.input_kwargs
        )
        with open(file := psiflow.context().new_file("orca_", ".inp"), "w") as f:
            f.write(input_str)
        return True, File(file)
