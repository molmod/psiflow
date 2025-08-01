import json
from functools import partial
from typing import Union

import typeguard
from parsl.app.app import bash_app, python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry, new_nullstate
from psiflow.reference.reference import Reference
from psiflow.utils.apps import copy_app_future
from psiflow.utils import TMP_COMMAND, CD_COMMAND


@typeguard.typechecked
def input_string(geometry: Geometry, gpaw_parameters: dict, properties: tuple) -> str:
    geometry_str = geometry.to_string()
    data = {
        "geometry": geometry_str,
        "gpaw_parameters": gpaw_parameters,
        "properties": properties,
    }
    return json.dumps(data)


def gpaw_singlepoint_pre(
    geometry: Geometry,
    gpaw_parameters: dict,
    properties: tuple,
    gpaw_command: str,
    parsl_resource_specification: dict = {},
    stdout: str = "",
    stderr: str = "",
) -> str:
    from psiflow.reference.gpaw_ import input_string

    input_str = input_string(geometry, gpaw_parameters, properties)
    write_command = f"echo '{input_str}' > input.json"
    command_list = [
        TMP_COMMAND,
        CD_COMMAND,
        write_command,
        gpaw_command,
    ]
    return "\n".join(command_list)


@typeguard.typechecked
def gpaw_singlepoint_post(
    geometry: Geometry,
    inputs: list = [],
) -> Geometry:
    with open(inputs[0], "r") as f:
        lines = f.read().split("\n")

    geometry = new_nullstate()  # GPAW parsing doesn't require initial geometry
    for i, line in enumerate(lines):
        if "CALCULATION SUCCESSFUL" in line:
            natoms = int(lines[i + 1])
            geometry_str = "\n".join(lines[i + 1 : i + 3 + natoms])
            geometry = Geometry.from_string(geometry_str)
            assert geometry.energy is not None
            geometry.stdout = inputs[0]
    return geometry


@typeguard.typechecked
@psiflow.serializable
class GPAW(Reference):
    outputs: list  # json does deserialize(serialize(tuple)) = list
    executor: str
    parameters: dict

    def __init__(
        self,
        outputs: Union[tuple, list] = ("energy", "forces"),
        executor: str = "GPAW",
        **parameters,
    ):
        self.outputs = list(outputs)
        self.parameters = parameters
        self.executor = executor
        self._create_apps()

    def _create_apps(self):
        definition = psiflow.context().definitions[self.executor]
        gpaw_command = definition.command()
        wq_resources = definition.wq_resources()
        app_pre = bash_app(gpaw_singlepoint_pre, executors=[self.executor])
        app_post = python_app(gpaw_singlepoint_post, executors=["default_threads"])
        self.app_pre = partial(
            app_pre,
            gpaw_parameters=self.parameters,
            properties=tuple(self.outputs),
            gpaw_command=gpaw_command,
            parsl_resource_specification=wq_resources,
        )
        self.app_post = app_post

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture:
        return copy_app_future(0.0)  # GPAW computes formation energy by default
