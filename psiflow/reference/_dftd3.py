import json
from functools import partial

import numpy as np
import typeguard
from parsl.app.app import bash_app, python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.reference.reference import Reference
from psiflow.utils.apps import copy_app_future
from psiflow.utils import TMP_COMMAND, CD_COMMAND


@typeguard.typechecked
def input_string(geometry: Geometry, parameters: dict, properties: tuple) -> str:
    geometry_str = geometry.to_string()
    data = {
        "geometry": geometry_str,
        "parameters": parameters,
        "properties": properties,
    }
    return json.dumps(data)


def d3_singlepoint_pre(
    geometry: Geometry,
    parameters: dict,
    properties: tuple,
    d3_command: str,
    stdout: str = "",
    stderr: str = "",
) -> str:
    from psiflow.reference._dftd3 import input_string
    input_str = input_string(geometry, parameters, properties)
    command_list = [
        TMP_COMMAND,
        CD_COMMAND,
        f"echo '{input_str}' > input.json",
        f"python -u {d3_command}",
    ]
    return "\n".join(command_list)


@typeguard.typechecked
def d3_singlepoint_post(
    geometry: Geometry,
    inputs: list = [],
) -> Geometry:
    from psiflow.geometry import new_nullstate

    with open(inputs[0], "r") as f:
        lines = f.read().split("\n")

    geometry = new_nullstate()
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
class D3(Reference):
    outputs: list  # json does deserialize(serialize(tuple)) = list
    executor: str
    parameters: dict

    def __init__(
        self,
        **parameters,
    ):
        self.parameters = parameters
        self.outputs = ["energy", "forces"]
        self.executor = "default_htex"
        self._create_apps()

    def _create_apps(self):
        path = "psiflow.reference._dftd3"
        d3_command = "$(python -c 'import {}; print({}.__file__)')".format(path, path)
        app_pre = bash_app(d3_singlepoint_pre, executors=["default_htex"])
        app_post = python_app(d3_singlepoint_post, executors=["default_threads"])
        self.app_pre = partial(
            app_pre,
            parameters=self.parameters,
            properties=tuple(self.outputs),
            d3_command=d3_command,
        )
        self.app_post = app_post

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture:
        return copy_app_future(0.0)  # GPAW computes formation energy by default


if __name__ == "__main__":
    from ase import Atoms
    from dftd3.ase import DFTD3

    with open("input.json", "r") as f:
        input_dict = json.loads(f.read())

    geometry = Geometry.from_string(input_dict["geometry"])
    parameters = input_dict["parameters"]
    properties = input_dict["properties"]

    atoms = Atoms(
        numbers=np.copy(geometry.per_atom.numbers),
        positions=np.copy(geometry.per_atom.positions),
        cell=np.copy(geometry.cell),
        pbc=geometry.periodic,
    )

    calculator = DFTD3(**parameters)
    atoms.calc = calculator

    if "forces" in properties:
        geometry.per_atom.forces[:] = atoms.get_forces()
    if "energy" in properties:
        geometry.energy = atoms.get_potential_energy()

    output_str = geometry.to_string()
    print("CALCULATION SUCCESSFUL")
    print(output_str)
