from __future__ import annotations  # necessary for type-guarding class methods

import logging
from functools import partial
from typing import Optional, Union
from pathlib import Path

import ase.symbols
import numpy as np
import typeguard
from ase.units import Bohr, Ha
from parsl.app.app import bash_app, python_app

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference
from psiflow.utils import TMP_COMMAND, CD_COMMAND
from psiflow.reference.utils import (
    find_line,
    Status,
    lines_to_array,
    copy_data_to_geometry,
)


logger = logging.getLogger(__name__)  # logging per module

# TODO: check_input?
# TODO: tests
# TODO: check for force_eval?
# TODO: check format_resources for units?
# TODO: check periodicity?


KEY_GHOST = "ghost"
KEY_DUMMY = 0
SYMBOLS = np.array(ase.symbols.chemical_symbols)
SYMBOLS[KEY_DUMMY] = "DA"


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


def format_geometry(
    geom: Geometry,
    charge: int = 0,
    multiplicity: int = 1,
) -> str:
    """"""
    symbols = SYMBOLS[geom.per_atom.numbers].tolist()
    # TODO: ghost atoms functionality?
    # if KEY_GHOST in atoms.arrays:
    #     for idx in np.flatnonzero(atoms.arrays[KEY_GHOST]):
    #         symbols[idx] = f"{symbols[idx]}:"
    header = f"*xyz {charge} {multiplicity}"
    pos = [
        f"{s:5} {p[0]:<15.8f} {p[1]:<15.8f} {p[2]:<15.8f}"
        for s, p in zip(symbols, geom.per_atom.positions)
    ]
    return "\n".join([header, *pos, "*"])


def format_resources(cores: int = 1, memory: int = 1000, **kwargs) -> str:
    mem = f"%maxcore {memory // cores}"
    cores = format_block("pal", [f"nprocs {cores}"])
    return "\n".join([mem, cores])


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
    input_template: str, geometry: Geometry, resources: dict, outputs: ()
) -> None:
    """"""
    xyz = format_geometry(geometry)
    res = format_resources(**resources)
    input_str = "\n".join([input_template, res, xyz])
    with open(outputs[0], "w") as f:
        f.write(input_str)


def _execute_calc(
    command: str = "",
    inputs: tuple = (),
    parsl_resource_specification: dict = None,
    stdout: str = "",
    stderr: str = "",
):
    """"""
    cp_command = f"cp {inputs[0].filepath} orca.inp"
    command_list = [TMP_COMMAND, CD_COMMAND, cp_command, command]
    return "\n".join(command_list)


def _process_output(
    geometry: Geometry,
    properties: tuple[str, ...],
    inputs: tuple[Path] = (),
) -> Geometry:
    """"""
    with open(inputs[0], "r") as f:
        stdout = f.read()
    try:
        data = parse_output(stdout, properties)
    except TypeError:
        # TODO: find out what went wrong
        data = None
    geometry = copy_data_to_geometry(geometry, data)
    geometry.order["stdout"] = inputs[0].name
    geometry.order["stderr"] = inputs[0].name
    return geometry


@typeguard.typechecked
@psiflow.serializable
class ORCA(Reference):
    outputs: Union[tuple, list]
    executor: str
    input_template: str

    def __init__(
        self,
        input_template: str,
        executor: str = "ORCA",
        outputs: Union[tuple, list] = ("energy", "forces"),
    ):
        self.executor = executor
        self.input_template = input_template
        self.outputs = list(outputs)
        self._create_apps()

    def app_pre(self, geometry, stdout, stderr):
        # TODO: temporary workaround until we overhaul everything
        future = self._app_pre(
            geometry=geometry, outputs=[psiflow.context().new_file("orca_", ".inp")]
        )
        future = self._app_execute(inputs=future.outputs, stdout=stdout, stderr=stderr)
        return future

    def _create_apps(self):
        definition = psiflow.context().definitions[self.executor]
        command = definition.command()
        wq_resources = definition.wq_resources()

        self._app_pre = partial(
            python_app(_create_input, executors=["default_threads"]),
            input_template=self.input_template,
            resources=wq_resources,
        )
        self._app_execute = partial(
            bash_app(_execute_calc, executors=[self.executor]),
            command=command,
            parsl_resource_specification=wq_resources,
        )
        self.app_post = partial(
            python_app(_process_output, executors=["default_threads"]),
            properties=self.outputs,
        )

    def get_single_atom_references(self, element):
        raise NotImplementedError
