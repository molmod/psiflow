from functools import partial
from collections.abc import Sequence

import numpy as np
from parsl import File, bash_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference, Status, _execute
from psiflow.utils.parse import find_line
from psiflow.utils.apps import copy_app_future


KEY_PARSE = 'CALCULATION SUCCESSFUL'

def make_bash_template() -> str:
    template = psiflow.context().bash_template
    command = f"echo '{KEY_PARSE}' && cat {{}} && echo '{KEY_PARSE}'"
    return template.format(commands=command, env="")


@psiflow.register_serializable
class ReferenceDummy(Reference):
    executor = "default_threads"
    _execute_label = "dummy_singlepoint"

    def __init__(self, outputs: Sequence[str] = ("energy", "forces")):
        self.outputs = tuple(outputs)
        self.bash_template = make_bash_template()

    def get_execute_app(self):
        # default_htex does not have an ExecutionDefinition
        return partial(
            bash_app(_execute, executors=[self.executor]),
            bash_template=self.bash_template,
            label=self._execute_label,
        )

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture:
        return copy_app_future(np.random.uniform())

    def create_input(self, geom: Geometry) -> tuple[bool, File]:
        with open(file := psiflow.context().new_file("dummy_", ".inp"), "w") as f:
            f.write(geom.to_string())
        return True, File(file)

    def parse_output(self, stdout: str) -> dict:
        lines = stdout.split("\n")
        idx_start = find_line(lines, KEY_PARSE) + 1
        idx_stop = find_line(lines, KEY_PARSE, idx_start)
        txt = "\n".join(lines[idx_start:idx_stop])
        geom = Geometry.from_string(txt)
        data = {
            "status": Status.SUCCESS,
            "positions": geom.per_atom.positions,
            "natoms": len(geom),
            "energy": np.random.uniform(),
        }
        if "forces" in self.outputs:
            data["forces"] = np.random.uniform(size=(len(geom), 3))
        return data
