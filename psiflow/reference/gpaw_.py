import json
import textwrap
from pathlib import Path
from collections.abc import Sequence

from parsl import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference, Status
from psiflow.utils.apps import copy_app_future
from psiflow.reference._gpaw import FILEPATH, DEFAULTS, STDOUT_KEY
from psiflow.utils.parse import find_line, format_env_vars


def make_bash_template(executor: str, script: str) -> str:
    context = psiflow.context()
    definition = context.definitions[executor]
    command = f"""
    cp {{}} input.json
    cp {script} script_gpaw.py
    {definition.command()}
    """
    command = textwrap.dedent(command)[1:]
    env = format_env_vars(definition.env_vars)
    return context.bash_template.format(commands=command, env=env)


def parse_output(stdout: str, properties: Sequence[str]) -> dict:
    lines = stdout.split("\n")
    idx_start = find_line(lines, STDOUT_KEY) + 1
    idx_stop = find_line(lines, STDOUT_KEY, idx_start)
    txt = "\n".join(lines[idx_start:idx_stop])
    geom = Geometry.from_string(txt)
    idx = find_line(lines, "Total:", reverse=True, max_lines=100)
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


@psiflow.register_serializable
class GPAW(Reference):
    executor: str = "GPAW"
    _execute_label: str = "gpaw_singlepoint"
    parameters: dict
    script: str

    def __init__(self, parameters: dict, script: str | Path = FILEPATH, **kwargs):
        super().__init__(**kwargs)
        self.parameters = parameters
        assert (script := Path(script)).is_file()
        self.script = str(script.resolve())  # absolute path
        self.bash_template = make_bash_template(self.executor, self.script)

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture:
        return copy_app_future(0.0)  # GPAW computes formation energy by default

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
