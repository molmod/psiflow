from __future__ import annotations  # necessary for type-guarding class methods

import logging
import warnings
import re
from functools import partial
from typing import Optional, Union
from pathlib import Path

import ase.symbols
import numpy as np
import parsl
from ase.units import Bohr, Ha
from parsl import bash_app, python_app, join_app, File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference
from psiflow.utils import TMP_COMMAND, CD_COMMAND
from psiflow.reference.utils import (
    find_line,
    Status,
    lines_to_array,
    copy_data_to_geometry,
    get_spin_multiplicities,
)
from psiflow.utils.apps import copy_app_future


logger = logging.getLogger(__name__)  # logging per module

# TODO: tests
# TODO: single_atom_reference


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
    # TODO: merge functionality? ghost atoms?
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
        data["runtime"] = lines[idx][16:]  # TODO: convert to number

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
    forces = lines_to_array(lines[idx_start:idx_stop], start=3, stop=6)
    data["forces"] = forces * Ha / Bohr

    return data


def _create_input(
    input_template: str, geom: Geometry, input_kwargs: dict, outputs: ()
) -> None:
    """"""
    input_str = input_template.format(coord=format_coord(geom), **input_kwargs)
    with open(outputs[0], "w") as f:
        f.write(input_str)


def _execute(
    file_in: File,
    command: str = "",
    parsl_resource_specification: dict = None,
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
    label: str = "orca_singlepoint",
) -> str:
    """"""
    cp_command = f"cp {file_in.filepath} orca.inp"
    command_list = [TMP_COMMAND, CD_COMMAND, cp_command, command]
    return "\n".join(command_list)


def _process_output(
    geom: Geometry,
    properties: tuple[str, ...],
    inputs: tuple[str] = (),
) -> Geometry:
    """"""
    # TODO: this one might be general for all Reference implementations
    # TODO: do we need properties?
    with open(inputs[0], "r") as f:
        stdout = f.read()
    try:
        data = parse_output(stdout, properties)
    except TypeError:
        # TODO: find out what went wrong
        data = {"status": Status.FAILED}
    data |= {"stdout": Path(inputs[0]).name, "stderr": Path(inputs[1]).name}
    return copy_data_to_geometry(geom, data)


@join_app
def evaluate(orca: ORCA, geom: Geometry) -> AppFuture[Geometry]:
    """"""
    if geom.periodic:
        msg = "ORCA does not support periodic boundary conditions, skipping geometry"
        warnings.warn(msg)
        return copy_app_future(geom)

    future = orca.app_pre(
        input_template=orca.input_template,
        geom=geom,
        input_kwargs=orca.input_kwargs,
        outputs=[psiflow.context().new_file("orca_", ".inp")],
    )
    future = orca.app_execute(file_in=future.outputs[0])
    future = orca.app_post(
        geom=geom,
        inputs=[future.stdout, future.stderr, future],  # wait for future
    )
    return future


@psiflow.serializable
class ORCA(Reference):
    outputs: tuple[str, ...]
    executor: str
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
        definition = psiflow.context().definitions[self.executor]
        command = definition.command()
        wq_resources = definition.wq_resources()
        self.app_pre = python_app(_create_input, executors=["default_threads"])
        self.app_execute = partial(
            bash_app(_execute, executors=[self.executor]),
            command=command,
            parsl_resource_specification=wq_resources,
        )
        self.app_post = partial(
            python_app(_process_output, executors=["default_threads"]),
            properties=self.outputs,
        )
        cores, memory = wq_resources["cores"], wq_resources["memory"]
        self.input_kwargs |= {"cores": cores, "memory": memory // cores}  # in MB

    def evaluate(self, geometry: Geometry | AppFuture) -> AppFuture:
        return evaluate(self, geometry)

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture[float]:
        assert box_size is None, "ORCA does not do PBC"
        return super().compute_atomic_energy(element)

    def get_single_atom_references(self, element: str) -> dict[int, Reference]:
        # TODO: this is not properly tested
        references = {}
        for mult in get_spin_multiplicities(element):
            ref = ORCA(
                self.input_template, outputs=self.outputs, executor=self.executor
            )
            ref.input_kwargs["multiplicity"] = mult
            references[mult] = ref
        return references
