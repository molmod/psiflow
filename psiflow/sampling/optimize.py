import warnings
import xml.etree.ElementTree as ET
from typing import Optional, Union

import parsl
from ase.units import Bohr, Ha
from parsl.app.app import bash_app, join_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.data.utils import write_frames
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Hamiltonian, MACEHamiltonian
from psiflow.sampling.sampling import (
    setup_sockets,
    make_server_command,
    make_driver_commands,
    make_wait_for_sockets_command,
)
from psiflow.sampling.output import HamiltonianComponent
from psiflow.utils.io import save_xml
from psiflow.utils.parse import format_env_vars


warnings.warn(
    "The 'optimize' module will likely be removed in a future release. "
    "Consider using the 'ase' module for structure optimisations instead.",
    FutureWarning,
)


def setup_forces(
    hamiltonian: Hamiltonian,
) -> tuple[list[HamiltonianComponent], ET.Element]:
    hamiltonian = 1.0 * hamiltonian  # convert to mixture
    forces = ET.Element("forces")
    components = []
    for n, h, c in zip(
        hamiltonian.get_named_components(),
        hamiltonian.hamiltonians,
        hamiltonian.coefficients,
    ):
        components.append(comp := HamiltonianComponent(n, h, True))
        force = ET.Element("force", forcefield=comp.name, weight=str(c))
        forces.append(force)
    return components, forces


def setup_motion(
    mode: str,
    etol: float,
    ptol: float,
    ftol: float,
) -> ET.Element:
    motion = ET.Element("motion", mode="minimize")
    optimizer = ET.Element("optimizer", mode=mode)
    tolerances = ET.Element("tolerances")

    energy = ET.Element("energy")
    energy.text = " {} ".format(etol / Ha)
    tolerances.append(energy)
    position = ET.Element("position")
    position.text = " {} ".format(ptol / Bohr)
    tolerances.append(position)
    force = ET.Element("force")
    force.text = " {} ".format(ftol / Ha * Bohr)
    tolerances.append(force)
    optimizer.append(tolerances)
    motion.append(optimizer)
    return motion


def setup_output(keep_trajectory: bool) -> ET.Element:
    output = ET.Element("output", prefix="output")
    checkpoint = ET.Element(
        "checkpoint",
        filename="checkpoint",
        stride="1",
        overwrite="True",
    )
    output.append(checkpoint)
    if keep_trajectory:
        trajectory = ET.Element(  # needed in any case
            "trajectory",
            stride="1",
            format="ase",
            filename="trajectory",
            bead="0",
        )
        trajectory.text = r" positions "
        output.append(trajectory)
    return output


def _execute_ipi(
    driver_kwargs: list[dict],
    command_server: str,
    env_vars: dict = {},
    bash_template: str = "",
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
    inputs: list = [],
    outputs: list = [],
    parsl_resource_specification: Optional[dict] = None,
) -> str:
    file_xml, file_xyz_in, *files_in = inputs
    command_start = make_server_command(
        command_server, file_xml, file_xyz_in, outputs[0], [], outputs[1:]
    )
    command_wait = make_wait_for_sockets_command(
        set(d["address"] for d in driver_kwargs)
    )
    commands_driver = make_driver_commands(driver_kwargs, file_xyz_in, files_in)
    command_list = [
        command_start,
        command_wait,
        *commands_driver,
        "wait",
    ]
    commands, env = "\n".join(command_list), format_env_vars(env_vars)
    return bash_template.format(commands=commands, env=env)


execute_ipi = bash_app(_execute_ipi, executors=["ModelEvaluation"])


def optimize(
    state: Union[Geometry, AppFuture],
    hamiltonian: Hamiltonian,
    steps: int = 5000,
    keep_trajectory: bool = False,
    mode: str = "lbfgs",
    etol: float = 1e-3,
    ptol: float = 1e-5,
    ftol: float = 1e-3,
) -> Union[AppFuture, tuple[AppFuture, Dataset]]:
    hamiltonian_components, forces = setup_forces(hamiltonian)
    sockets = setup_sockets(hamiltonian_components)

    initialize = ET.Element("initialize", nbeads="1")
    start = ET.Element("file", mode="ase", cell_units="angstrom")
    start.text = " start_0.xyz "
    initialize.append(start)
    motion = setup_motion(mode, etol, ptol, ftol)

    system = ET.Element("system", prefix="walker-0")
    system.append(initialize)
    system.append(motion)
    system.append(forces)

    output = setup_output(keep_trajectory)

    simulation = ET.Element("simulation", mode="static")
    simulation.append(output)
    for socket in sockets:
        simulation.append(socket)
    simulation.append(system)
    total_steps = ET.Element("total_steps")
    total_steps.text = " {} ".format(steps)
    simulation.append(total_steps)

    context = psiflow.context()
    definition = context.definitions["ModelEvaluation"]
    input_future = save_xml(
        simulation,
        outputs=[context.new_file("input_", ".xml")],
    ).outputs[0]
    inputs = [
        input_future,
        Dataset([state]).extxyz,
    ]
    outputs = [context.new_file("data_", ".xyz")]
    if keep_trajectory:
        outputs.append(context.new_file("opt_", ".xyz"))

    driver_kwargs = []
    for i, comp in enumerate(hamiltonian_components):
        inputs.append(comp.hamiltonian.serialize_function(dtype="float64"))
        kwargs = {"idx": i, "address": comp.address}
        if isinstance(comp.hamiltonian, MACEHamiltonian):
            kwargs |= definition.get_driver_resources.get(1, 1)[0]
        driver_kwargs.append(kwargs)

    result = execute_ipi(
        driver_kwargs,
        definition.server_command(),
        env_vars=definition.env_vars,
        bash_template=context.bash_template,
        inputs=inputs,
        outputs=outputs,
        parsl_resource_specification=definition.wq_resources(1),
    )

    final = Dataset(None, result.outputs[0]).evaluate(hamiltonian)[-1]
    if keep_trajectory:
        trajectory = Dataset(None, result.outputs[1])
        return final, trajectory
    else:
        return final


@join_app
def _optimize_dataset(
    geometries: list[Geometry], *args, outputs: list = [], **kwargs
) -> AppFuture:
    assert not kwargs.get("keep_trajectory", False)
    optimized = []
    for geometry in geometries:
        optimized.append(optimize(geometry, *args, **kwargs))
    return write_frames(*optimized, outputs=[outputs[0]])


def optimize_dataset(dataset: Dataset, *args, **kwargs) -> Dataset:
    extxyz = _optimize_dataset(
        dataset.geometries(),
        *args,
        outputs=[psiflow.context().new_file("data_", ".xyz")],
        **kwargs,
    ).outputs[0]
    return Dataset(None, extxyz)
