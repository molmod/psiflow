from __future__ import annotations  # necessary for type-guarding class methods

import xml.etree.ElementTree as ET
from typing import Optional

import parsl
import typeguard
from parsl.app.app import bash_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, Geometry
from psiflow.hamiltonians.hamiltonian import Hamiltonian
from psiflow.tools.optimize import setup_forces, setup_sockets
from psiflow.utils import load_numpy, save_xml


@typeguard.typechecked
def setup_motion(
    mode: str,
    asr: str,
    pos_shift: float,
    energy_shift: float,
) -> ET.Element:
    motion = ET.Element("motion", mode="vibrations")
    vibrations = ET.Element("vibrations", mode="fd")
    pos = ET.Element("pos_shift")
    pos.text = " {} ".format(pos_shift)
    vibrations.append(pos)
    energy = ET.Element("energy_shift")
    energy.text = " {} ".format(energy_shift)
    vibrations.append(energy)
    prefix = ET.Element("prefix")
    prefix.text = " output "
    vibrations.append(prefix)
    asr_ = ET.Element("asr")
    asr_.text = " {} ".format(asr)
    vibrations.append(asr_)
    motion.append(vibrations)
    return motion


def _execute_ipi(
    hamiltonian_names: list[str],
    client_args: list[str],
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
        address = name.lower()
        command_ = command_client + " --address={}".format(address)
        command_ += " --path_hamiltonian={}".format(inputs[2 + i].filepath)
        command_ += " --start={}".format(inputs[1].filepath)
        command_ += " " + client_args[i] + " "
        command_ += " & \n"
        command_clients += command_

    command_end = command_server
    command_end += " --cleanup;"
    command_copy = " cp i-pi.output_full.hess {};".format(outputs[0])
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
def compute_harmonic(
    state: Geometry,
    hamiltonian: Hamiltonian,
    mode: str = "fd",
    asr: str = "crystal",
    pos_shift: float = 0.01,
    energy_shift: float = 0.00095,
) -> AppFuture:
    hamiltonians_map, forces = setup_forces(hamiltonian)
    sockets = setup_sockets(hamiltonians_map)

    initialize = ET.Element("initialize", nbeads="1")
    start = ET.Element("file", mode="ase", cell_units="angstrom")
    start.text = " start_0.xyz "
    initialize.append(start)
    motion = setup_motion(mode, asr, pos_shift, energy_shift)

    system = ET.Element("system")
    system.append(initialize)
    system.append(motion)
    system.append(forces)

    # output = setup_output(keep_trajectory)

    simulation = ET.Element("simulation", mode="static")
    # simulation.append(output)
    for socket in sockets:
        simulation.append(socket)
    simulation.append(system)
    total_steps = ET.Element("total_steps")
    total_steps.text = " {} ".format(1000000)
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
        nclients, args = definition.get_client_args(name, 1)
        client_args.append(args)
    outputs = [
        context.new_file("hess_", ".txt"),
    ]

    command_server = definition.server_command()
    command_client = definition.client_command()
    resources = definition.wq_resources(1)

    result = execute_ipi(
        hamiltonian_names,
        client_args,
        command_server,
        command_client,
        stdout=parsl.AUTO_LOGNAME,
        stderr=parsl.AUTO_LOGNAME,
        inputs=inputs,
        outputs=outputs,
        parsl_resource_specification=resources,
    )
    return load_numpy(inputs=[result.outputs[0]])
