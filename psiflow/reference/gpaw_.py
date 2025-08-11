from __future__ import annotations  # necessary for type-guarding class methods

import json
from typing import Optional, Union

from parsl import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference
from psiflow.utils import TMP_COMMAND, CD_COMMAND
from psiflow.reference.utils import Status, find_line
from psiflow.utils.apps import copy_app_future
from psiflow.reference._gpaw import FILEPATH, DEFAULTS, STDOUT_KEY


def parse_output(stdout: str, properties: tuple[str, ...]) -> dict:
    lines = stdout.split("\n")
    idx_start = find_line(lines, STDOUT_KEY) + 1
    idx_stop = find_line(lines, STDOUT_KEY, idx_start)
    txt = "\n".join(lines[idx_start:idx_stop])
    geom = Geometry.from_string(txt)
    idx = find_line(lines, "Total:", reverse=True)
    data = {
        "status": Status.SUCCESS,
        "runtime": lines[idx].split()[-2],
        "positions": geom.per_atom.positions,
        "natoms": len(geom),
        "energy": geom.energy,
    }
    if "forces" in properties:
        data["forces"] = geom.per_atom.forces
    return data


@psiflow.serializable
class GPAW(Reference):
    _execute_label = "gpaw_singlepoint"

    def __init__(
        self,
        parameters: dict,
        executable: str = FILEPATH,
        outputs: Union[tuple, list] = ("energy", "forces"),
        executor: str = "GPAW",
    ):
        self.outputs = tuple(outputs)
        self.parameters = parameters
        self.executable = executable
        self.executor = executor
        self._create_apps()

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture:
        return copy_app_future(0.0)  # GPAW computes formation energy by default

    def get_shell_command(self, inputs: list[File]) -> str:
        command_list = [
            TMP_COMMAND,
            CD_COMMAND,
            f"cp {inputs[0].filepath} input.json",
            f"cp {self.executable} script_gpaw.py",
            self.execute_command,
        ]
        return "\n".join(command_list)

    def parse_output(self, stdout: str) -> dict:
        return parse_output(stdout, self.outputs)

    def create_input(self, geom: Geometry) -> tuple[bool, File]:
        data = {
            "geometry": geom.to_string(),
            "gpaw_parameters": self.parameters,
            "properties": self.outputs,
        }
        with open(file := psiflow.context().new_file("gpaw_", ".json"), "w") as f:
            f.write(json.dumps(data))
        return True, File(file)
