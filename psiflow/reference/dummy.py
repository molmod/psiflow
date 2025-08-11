from __future__ import annotations  # necessary for type-guarding class methods

from typing import Optional, Union
from functools import partial

import numpy as np
from parsl import File, bash_app, python_app

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference, Status
from psiflow.reference.reference import _execute, _process_output


@psiflow.serializable
class ReferenceDummy(Reference):
    _execute_label = "dummy_singlepoint"

    def __init__(self, outputs: Union[tuple, list] = ("energy", "forces")):
        self.outputs = outputs
        self._create_apps()

    def _create_apps(self):
        # psiflow.context().definitions does not contain "default_htex"
        self.execute_command = ""
        self.app_pre = self.create_input
        self.app_execute = partial(
            bash_app(_execute, executors=["default_htex"]),
            reference=self,
            parsl_resource_specification={},
            label=self._execute_label,
        )
        self.app_post = partial(
            python_app(_process_output, executors=["default_threads"]),
            reference=self,
        )

    def get_shell_command(self, inputs: list[File]) -> str:
        return f"cat {inputs[0].filepath}"

    def parse_output(self, stdout: str) -> dict:
        geom = Geometry.from_string(stdout)
        data = {
            "status": Status.SUCCESS,
            "runtime": np.nan,
            "positions": geom.per_atom.positions,
            "natoms": len(geom),
            "energy": np.random.uniform(),
        }
        if "forces" in self.outputs:
            data["forces"] = np.random.uniform(size=(len(geom), 3))
        return data

    def create_input(self, geom: Geometry) -> tuple[bool, File]:
        with open(file := psiflow.context().new_file("dummy_", ".inp"), "w") as f:
            f.write(geom.to_string())
        return True, File(file)
