from functools import partial

import numpy as np
from parsl import File, bash_app

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference, Status, _execute


def make_bash_template() -> str:
    template = psiflow.context().bash_template
    return template.format(commands="cat {}", env="")


@psiflow.register_serializable
class ReferenceDummy(Reference):
    executor = "default_threads"
    _execute_label = "dummy_singlepoint"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bash_template = make_bash_template()

    def get_execute_app(self):
        # default_threads does not have an ExecutionDefinition
        return partial(
            bash_app(_execute, executors=[self.executor]),
            bash_template=self.bash_template,
            label=self._execute_label,
        )

    def create_input(self, geom: Geometry) -> tuple[bool, File]:
        with open(file := psiflow.context().new_file("dummy_", ".inp"), "w") as f:
            f.write(geom.to_string())
        return True, File(file)

    def parse_output(self, stdout: str) -> dict:
        geom = Geometry.from_string(stdout)
        data = {
            "status": Status.SUCCESS,
            "positions": geom.per_atom.positions,
            "natoms": len(geom),
            "energy": np.random.uniform(),
        }
        if "forces" in self.outputs:
            data["forces"] = np.random.uniform(size=(len(geom), 3))
        return data
