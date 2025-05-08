from __future__ import annotations  # necessary for type-guarding class methods

import xml.etree.ElementTree as ET
from typing import Optional, Union

import parsl
import typeguard
from ase.units import Bohr, Ha
from parsl.app.app import bash_app, join_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.data.utils import write_frames
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Hamiltonian
from psiflow.sampling.sampling import setup_sockets, make_start_command, make_client_command
from psiflow.utils.io import save_xml
from psiflow.utils import TMP_COMMAND, CD_COMMAND


@typeguard.typechecked
def setup_forces(hamiltonian: Hamiltonian) -> tuple[dict[str, Hamiltonian], ET.Element]:
    hamiltonian = 1.0 * hamiltonian  # convert to mixture
    counts = {}
    hamiltonians_map = {}
    forces = ET.Element("forces")
    for h, c in zip(hamiltonian.hamiltonians, hamiltonian.coefficients):
        name = h.__class__.__name__
        if name not in counts:
            counts[name] = 0
        count = counts.get(name)
        counts[name] += 1
        force = ET.Element("force", forcefield=name + str(count), weight=str(c))
        forces.append(force)
        hamiltonians_map[name + str(count)] = h
    return hamiltonians_map, forces


@typeguard.typechecked
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


@typeguard.typechecked
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
    hamiltonian_names: list[str],
    client_args: list[list[str]],
    keep_trajectory: bool,
    command_server: str,
    command_client: str,
    env_vars: dict = {},
    stdout: str = "",
    stderr: str = "",
    inputs: list = [],
    outputs: list = [],
    parsl_resource_specification: Optional[dict] = None,
) -> str:
    env_command = 'export ' + ' '.join([f"{name}={value}" for name, value in env_vars.items()])
    command_start = make_start_command(command_server, inputs[0], inputs[1])
    commands_client = []
    for i, name in enumerate(hamiltonian_names):
        args = client_args[i]
        assert len(args) == 1  # only have one client per hamiltonian
        for arg in args:
            commands_client += make_client_command(command_client, name, inputs[2 + i], inputs[1], arg),

    command_end = f'{command_server} --cleanup --output_xyz={outputs[0].filepath}'
    command_copy = f'cp walker-0_output.trajectory_0.ase {outputs[1].filepath}' if keep_trajectory else ''
    command_list = [
        TMP_COMMAND,
        CD_COMMAND,
        env_command,
        command_start,
        *commands_client,
        "wait",
        command_end,
        command_copy,
    ]
    return "\n".join(command_list)


execute_ipi = bash_app(_execute_ipi, executors=["ModelEvaluation"])


@typeguard.typechecked
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
    hamiltonians_map, forces = setup_forces(hamiltonian)
    sockets = setup_sockets(hamiltonians_map)

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
    inputs += [h.serialize_function(dtype="float64") for h in hamiltonians_map.values()]

    hamiltonian_names = list(hamiltonians_map.keys())
    client_args = []
    for name in hamiltonian_names:
        args = definition.get_client_args(name, 1, "minimize")
        client_args.append(args)
    outputs = [context.new_file("data_", ".xyz")]
    if keep_trajectory:
        outputs.append(context.new_file("opt_", ".xyz"))

    command_server = definition.server_command()
    command_client = definition.client_command()
    resources = definition.wq_resources(1)

    result = execute_ipi(
        hamiltonian_names,
        client_args,
        keep_trajectory,
        command_server,
        command_client,
        env_vars=definition.env_vars,
        stdout=parsl.AUTO_LOGNAME,
        stderr=parsl.AUTO_LOGNAME,
        inputs=inputs,
        outputs=outputs,
        parsl_resource_specification=resources,
    )

    final = Dataset(None, result.outputs[0]).evaluate(hamiltonian)[-1]
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
