from __future__ import annotations  # necessary for type-guarding class methods

import xml.etree.ElementTree as ET
from typing import Optional, Union

import parsl
import typeguard
from ase.units import Bohr, Ha
from parsl.app.app import bash_app, join_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, write_frames
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Hamiltonian
from psiflow.utils import save_xml


@typeguard.typechecked
def setup_sockets(
    hamiltonians_map: dict[str, Hamiltonian],
) -> list[ET.Element]:
    sockets = []
    for name in hamiltonians_map.keys():
        ffsocket = ET.Element("ffsocket", mode="unix", name=name, pbc="False")
        timeout = ET.Element("timeout")
        timeout.text = str(
            60 * psiflow.context().definitions["ModelEvaluation"].timeout
        )
        ffsocket.append(timeout)
        exit_on = ET.Element("exit_on_disconnect")
        exit_on.text = " TRUE "
        ffsocket.append(exit_on)
        address = ET.Element("address")  # placeholder
        address.text = name.lower()
        ffsocket.append(address)

        sockets.append(ffsocket)
    return sockets


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
        )
        output.append(trajectory)
    return output


def _execute_ipi(
    hamiltonian_names: list[str],
    client_args: list[list[str]],
    keep_trajectory: bool,
    command_server: str,
    command_client: str,
    stdout: str = "",
    stderr: str = "",
    inputs: list = [],
    outputs: list = [],
    parsl_resource_specification: Optional[dict] = None,
) -> str:
    tmp_command = "tmpdir=$(mktemp -d);"
    cd_command = "cd $tmpdir;"
    command_start = command_server + " --nwalkers=1"
    command_start += " --input_xml={}".format(inputs[0].filepath)
    command_start += " --start_xyz={}".format(inputs[1].filepath)
    command_start += "  & \n"
    command_clients = ""
    for i, name in enumerate(hamiltonian_names):
        args = client_args[i]
        assert len(args) == 1  # only have one client per hamiltonian
        for _j, arg in enumerate(args):
            command_ = command_client + " --address={}".format(name.lower())
            command_ += " --path_hamiltonian={}".format(inputs[2 + i].filepath)
            command_ += " --start={}".format(inputs[1].filepath)
            command_ += " " + arg + " "
            command_ += " & \n"
            command_clients += command_

    command_end = command_server
    command_end += " --cleanup"
    command_end += " --output_xyz={}; ".format(outputs[0].filepath)
    command_copy = ""
    if keep_trajectory:
        command_copy += "cp walker-0_output.trajectory_0.ase {}; ".format(
            outputs[1].filepath
        )
    command_list = [
        tmp_command,
        cd_command,
        command_start,
        "sleep 3s;",
        command_clients,
        "wait;",
        command_end,
        command_copy,
    ]
    return " ".join(command_list)


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
    inputs += [h.serialize_calculator() for h in hamiltonians_map.values()]

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
        stdout=parsl.AUTO_LOGNAME,
        stderr=parsl.AUTO_LOGNAME,
        inputs=inputs,
        outputs=outputs,
        parsl_resource_specification=resources,
    )

    trajectory = Dataset(None, result.outputs[0])
    final = hamiltonian.evaluate(trajectory[-1:])[0]
    if keep_trajectory:
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
