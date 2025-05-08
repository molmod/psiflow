from __future__ import annotations  # necessary for type-guarding class methods

from typing import Optional, Union

import parsl
import typeguard
from parsl.app.app import bash_app, join_app
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.data import Dataset
from psiflow.data.utils import write_frames
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Hamiltonian
from psiflow.utils.io import dump_json
from psiflow.sampling.sampling import serialize_mixture, label_forces
from psiflow.utils import TMP_COMMAND, CD_COMMAND

from ._ase import ALLOWED_MODES

EXECUTABLE = 'psiflow-ase-opt'      # not stored in ModelEvaluation (yet?)


def _execute_ase(
    command_launch: str,
    inputs: list[DataFuture],
    outputs: list[DataFuture],
    env_vars: dict = {},
    stdout: str = "",
    stderr: str = "",
    parsl_resource_specification: Optional[dict] = None,
) -> str:
    env_command = 'export ' + ' '.join([f"{name}={value}" for name, value in env_vars.items()])
    command_start = ' '.join([
        f'{command_launch} run --input_config={inputs[0].filepath} --start_xyz={inputs[1].filepath}',
        *[f'--path_hamiltonian={future.filepath}' for future in inputs[2:]], '&'
    ])
    command_end = f'{command_launch} clean --output_xyz={outputs[0].filepath}'
    if len(outputs) == 2:
        command_end += f' --output_traj={outputs[1].filepath}'

    command_list = [
        TMP_COMMAND,
        CD_COMMAND,
        env_command,
        command_start,
        "wait",
        command_end,
    ]
    return "\n".join(command_list)


execute_ase = bash_app(_execute_ase, executors=["ModelEvaluation"])


@typeguard.typechecked
def optimize(
    state: Union[Geometry, AppFuture],
    hamiltonian: Hamiltonian,
    mode: str = 'full',
    steps: int = int(1e12),
    keep_trajectory: bool = False,
    pressure: float = 0,
    f_max: float = 1e-3,
) -> Union[AppFuture, tuple[AppFuture, Dataset]]:

    assert mode in ALLOWED_MODES
    assert steps > 0
    assert f_max > 0

    context = psiflow.context()
    definition = context.definitions["ModelEvaluation"]

    command_list = [EXECUTABLE]
    if definition.max_simulation_time is not None:
        max_time = 0.9 * (60 * definition.max_simulation_time)
        command_list = ["timeout -s 15 {}s".format(max_time), *command_list]
    command_launch = " ".join(command_list)

    input_geometry = Dataset([state]).extxyz
    hamiltonian = 1.0 * hamiltonian  # convert to mixture
    names, coeffs = label_forces(hamiltonian), hamiltonian.coefficients
    input_forces = serialize_mixture(hamiltonian, dtype="float64")          # double precision for MLPs
    forces = [
        dict(forcefield=n, weight=str(c), file=f.filename) for n, c, f in zip(names, coeffs, input_forces)
    ]

    config = dict(
        task='ASE optimisation',
        forces=forces,
        mode=mode,
        f_max=f_max,
        pressure=pressure,
        max_steps=steps,
        keep_trajectory=keep_trajectory,
    )
    input_future = dump_json(
        outputs=[context.new_file("input_", ".json")],
        **config,
    ).outputs[0]
    inputs = [input_future, input_geometry, *input_forces]

    outputs = [context.new_file("data_", ".xyz")]
    if keep_trajectory:
        outputs.append(context.new_file("opt_", ".xyz"))

    result = execute_ase(
        command_launch=command_launch,
        env_vars=definition.env_vars,
        inputs=inputs,
        outputs=outputs,
        stdout=parsl.AUTO_LOGNAME,
        stderr=parsl.AUTO_LOGNAME,
        parsl_resource_specification=definition.wq_resources(1),
    )

    final = Dataset(None, result.outputs[0])[-1]
    if keep_trajectory:
        trajectory = Dataset(None, result.outputs[1])
        return final, trajectory
    else:
        return final


@join_app
@typeguard.typechecked
def _optimize_dataset(
    geometries: list[Geometry], *args, outputs: list = [], **kwargs
) -> AppFuture:
    assert not kwargs.get("keep_trajectory", False)
    optimized = []
    for geometry in geometries:
        optimized.append(optimize(geometry, *args, **kwargs))
    return write_frames(*optimized, outputs=[outputs[0]])


@typeguard.typechecked
def optimize_dataset(dataset: Dataset, *args, **kwargs) -> Dataset:
    extxyz = _optimize_dataset(
        dataset.geometries(),
        *args,
        outputs=[psiflow.context().new_file("data_", ".xyz")],
        **kwargs,
    ).outputs[0]
    return Dataset(None, extxyz)
