import json
from functools import partial
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import bash_app, join_app, python_app
from parsl.app.bash import BashApp
from parsl.app.python import PythonApp
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry, new_nullstate
from psiflow.reference.reference import Reference
from psiflow.utils import copy_app_future


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
    parsl_resource_specification: Optional[dict] = None,
    stdout: str = "",
    stderr: str = "",
) -> str:
    from psiflow.reference.gpaw_ import input_string

    input_str = input_string(geometry, gpaw_parameters, properties)
    tmp_command = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    cd_command = "cd $mytmpdir;"
    write_command = "echo '{}' > input.json;".format(input_str)
    command_list = [
        tmp_command,
        cd_command,
        write_command,
        gpaw_command,
    ]
    return " ".join(command_list)


@typeguard.typechecked
def gpaw_singlepoint_post(inputs: list = []) -> Geometry:
    with open(inputs[0], "r") as f:
        lines = f.read().split("\n")

    geometry = None
    for i, line in enumerate(lines):
        if "CALCULATION SUCCESSFUL" in line:
            natoms = int(lines[i + 1])
            geometry_str = "\n".join(lines[i + 1 : i + 3 + natoms])
            geometry = Geometry.from_string(geometry_str)
            assert geometry.energy is not None
    if geometry is None:
        geometry = new_nullstate()
    geometry.stdout = inputs[0]
    return geometry


@join_app
@typeguard.typechecked
def evaluate_single(
    geometry: Union[Geometry, AppFuture],
    gpaw_parameters: dict,
    properties: tuple,
    gpaw_command: str,
    wq_resources: dict[str, Union[float, int]],
    app_pre: BashApp,
    app_post: PythonApp,
) -> AppFuture:
    import parsl

    from psiflow.geometry import NullState
    from psiflow.utils import copy_app_future

    if geometry == NullState:
        return copy_app_future(NullState)
    else:
        pre = app_pre(
            geometry,
            gpaw_parameters,
            properties,
            gpaw_command=gpaw_command,
            stdout=parsl.AUTO_LOGNAME,
            stderr=parsl.AUTO_LOGNAME,
            parsl_resource_specification=wq_resources,
        )
        return app_post(
            inputs=[pre.stdout, pre.stderr, pre],  # wait for bash app
        )


@typeguard.typechecked
@psiflow.serializable
class GPAW(Reference):
    properties: list[str]  # json does deserialize(serialize(tuple)) = list
    parameters: dict

    def __init__(
        self,
        properties: Union[tuple, list] = ("energy", "forces"),
        **parameters,
    ):
        self.properties = list(properties)
        self.parameters = parameters
        self._create_apps()

    def _create_apps(self):
        definition = psiflow.context().definitions["ReferenceEvaluation"]
        gpaw_command = definition.gpaw_command()
        executor_label = definition.name
        wq_resources = definition.wq_resources()
        app_pre = bash_app(gpaw_singlepoint_pre, executors=[executor_label])
        app_post = python_app(gpaw_singlepoint_post, executors=["default_threads"])
        self.evaluate_single = partial(
            evaluate_single,
            gpaw_parameters=self.parameters,
            properties=tuple(self.properties),
            gpaw_command=gpaw_command,
            wq_resources=wq_resources,
            app_pre=app_pre,
            app_post=app_post,
        )

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture:
        return copy_app_future(0.0)  # GPAW computes formation energy by default


if __name__ == "__main__":
    from ase import Atoms
    from ase.calculators.mixing import SumCalculator
    from dftd3.ase import DFTD3
    from gpaw import GPAW as GPAWCalculator

    def minimal_box(
        atoms: Atoms,
        border: float = 0.0,
        h: float = 0.2,
        multiple: int = 4,
    ) -> None:
        # inspired by gpaw.cluster.Cluster
        if len(atoms) == 0:
            return None
        min_bounds, max_bounds = np.array(
            [np.minimum.reduce(atoms.positions), np.maximum.reduce(atoms.positions)]
        )
        if isinstance(border, list):
            b = np.array(border)
        else:
            b = np.array([border, border, border])
        if not hasattr(h, "__len__"):
            h = np.array([h, h, h])
        min_bounds -= b
        max_bounds += b - min_bounds
        grid_points = np.ceil(max_bounds / h / multiple) * multiple
        length_diff = grid_points * h - max_bounds
        max_bounds += length_diff
        min_bounds -= length_diff / 2
        shift = tuple(-1.0 * min_bounds)
        atoms.translate(shift)
        atoms.set_cell(tuple(max_bounds))

    with open("input.json", "r") as f:
        input_dict = json.loads(f.read())

    geometry = Geometry.from_string(input_dict["geometry"])
    gpaw_parameters = input_dict["gpaw_parameters"]
    properties = input_dict["properties"]
    d3 = gpaw_parameters.pop("d3", {})

    atoms = Atoms(
        numbers=np.copy(geometry.per_atom.numbers),
        positions=np.copy(geometry.per_atom.positions),
        cell=np.copy(geometry.cell),
        pbc=geometry.periodic,
    )
    if not geometry.periodic:
        minimal_box(
            atoms,
            gpaw_parameters.get("h", 0.2),
            gpaw_parameters.pop("minimal_box_border", 2),  # if present, remove
            gpaw_parameters.pop("minimal_box_multiple", 4),
        )

    calculator = GPAWCalculator(**gpaw_parameters)
    if len(d3) > 0:
        calculator = SumCalculator([calculator, DFTD3(**d3)])
    atoms.calc = calculator

    if "forces" in properties:
        geometry.per_atom.forces[:] = atoms.get_forces()
    if "energy" in properties:
        geometry.energy = atoms.get_potential_energy()

    output_str = geometry.to_string()
    print("CALCULATION SUCCESSFUL")
    print(output_str)
