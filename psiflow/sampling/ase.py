from typing import Optional, Union

import parsl
from parsl import bash_app, join_app, python_app, File
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.data import Dataset
from psiflow.data.utils import write_frames
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Hamiltonian
from psiflow.utils.apps import setup_logger
from psiflow.utils.io import _dump_json
from psiflow.utils.parse import get_task_name_id
from psiflow.execution import format_env_vars

from ._ase import ALLOWED_MODES, __file__ as file_ase

DEFAULT_EXECUTABLE = "script.py"
logger = setup_logger(__name__)  # logging per module


class OptimisationFailedError(Exception):
    pass


@python_app(executors=["default_threads"])
def check_task_output(
    file: DataFuture, file_stdout: str, file_stderr: str
) -> AppFuture:
    # TODO: find actual reason for fail?
    task_name, task_id = get_task_name_id(file_stdout)
    try:
        geom = Geometry.load(file.filepath)
        status = "SUCCESS"
    except Exception as e:
        # output file is empty because the optimisation failed
        status = "FAILED"

    logger.info(f'Task "{task_name}" (ID {task_id}): {status}')
    if status == "SUCCESS":
        return geom
    raise OptimisationFailedError(f"Task {task_id}")


def _execute_ase(
    command_launch: str,
    inputs: list[DataFuture],
    outputs: list[DataFuture],
    env_vars: dict = {},
    bash_template: str = "",
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
    parsl_resource_specification: Optional[dict] = None,
) -> str:
    command_opt_args = [
        command_launch,
        f"--input_config={inputs[1].filepath}",
        f"--start_xyz={inputs[2].filepath}",
        *[f"--path_hamiltonian={future.filepath}" for future in inputs[3:]],
        f"--output_xyz={outputs[0].filepath}",
    ]
    if len(outputs) == 2:
        command_opt_args.append(f"--output_traj={outputs[1].filepath}")

    command_list = [
        f"cp {inputs[0].filepath} {DEFAULT_EXECUTABLE}",
        " ".join(command_opt_args),
    ]
    commands, env = "\n".join(command_list), format_env_vars(env_vars)
    return bash_template.format(commands=commands, env=env)


execute_ase = bash_app(_execute_ase, executors=["ModelEvaluation"])


def optimize(
    state: Union[Geometry, AppFuture],
    hamiltonian: Hamiltonian,
    mode: str = "full",
    steps: int = int(1e12),
    keep_trajectory: bool = False,
    pressure: float = 0,
    f_max: float = 1e-3,
    script: str = file_ase,
) -> Union[AppFuture, tuple[AppFuture, Dataset]]:

    assert mode in ALLOWED_MODES
    assert steps > 0
    assert f_max > 0

    context = psiflow.context()
    definition = context.definitions["ModelEvaluation"]
    command = f"python -u {DEFAULT_EXECUTABLE}"
    command_launch = definition.wrap_in_timeout(command)

    input_geometry = Dataset([state]).extxyz  # state can be future
    hamiltonian = 1.0 * hamiltonian  # convert to mixture
    names, coeffs = hamiltonian.get_named_components(), hamiltonian.coefficients
    input_forces = hamiltonian.serialize(dtype="float64")  # double precision for MLPs
    forces = [
        dict(forcefield=n, weight=str(c), file=f.filename)
        for n, c, f in zip(names, coeffs, input_forces)
    ]
    config = dict(
        task="ASE optimisation",
        forces=forces,
        mode=mode,
        f_max=f_max,
        pressure=pressure,
        max_steps=steps,
        keep_trajectory=keep_trajectory,
    )
    file_config = context.new_file("input_", ".json")
    _dump_json(outputs=[file_config], **config)

    inputs = [File(script), file_config, input_geometry, *input_forces]
    outputs = [context.new_file("data_", ".xyz")]
    if keep_trajectory:
        outputs.append(context.new_file("opt_", ".xyz"))

    result = execute_ase(
        command_launch=command_launch,
        env_vars=definition.env_vars,
        bash_template=context.bash_template,
        inputs=inputs,
        outputs=outputs,
        parsl_resource_specification=definition.wq_resources(1),
    )

    geom = check_task_output(result.outputs[0], result.stdout, result.stderr)
    if keep_trajectory:
        trajectory = Dataset(None, result.outputs[1])
        return geom, trajectory
    else:
        return geom


@join_app
def _optimize_dataset(
    geometries: list[Geometry], *args, outputs: list = [], **kwargs
) -> AppFuture:
    assert not kwargs.get("keep_trajectory", False)
    logger.info(f"Performing {len(geometries)} structure optimisations.")
    optimized = [optimize(geometry, *args, **kwargs) for geometry in geometries]
    return write_frames(*optimized, outputs=[outputs[0]])


def optimize_dataset(dataset: Dataset, *args, **kwargs) -> Dataset:
    extxyz = _optimize_dataset(
        dataset.geometries(),
        *args,
        outputs=[psiflow.context().new_file("data_", ".xyz")],
        **kwargs,
    ).outputs[0]
    return Dataset(None, extxyz)
