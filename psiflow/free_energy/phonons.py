from __future__ import annotations  # necessary for type-guarding class methods

import xml.etree.ElementTree as ET
from typing import Optional, Union

import numpy as np
import parsl
import typeguard
from ase.units import Bohr, Ha, J, _c, _hplanck, _k, kB, second
from parsl.app.app import bash_app, python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry, mass_weight
from psiflow.hamiltonians import Hamiltonian, MixtureHamiltonian
from psiflow.sampling.sampling import (
    setup_sockets,
    label_forces,
    make_force_xml,
    serialize_mixture,
    make_start_command,
    make_client_command
)
from psiflow.utils.apps import multiply
from psiflow.utils.io import load_numpy, save_xml
from psiflow.utils import TMP_COMMAND, CD_COMMAND


@typeguard.typechecked
def _compute_frequencies(hessian: np.ndarray, geometry: Geometry) -> np.ndarray:
    assert hessian.shape[0] == hessian.shape[1]
    assert len(geometry) * 3 == hessian.shape[0]
    return np.sqrt(np.linalg.eigvalsh(mass_weight(hessian, geometry))) / (2 * np.pi)


compute_frequencies = python_app(_compute_frequencies, executors=["default_threads"])


@typeguard.typechecked
def _harmonic_free_energy(
    frequencies: Union[float, np.ndarray],
    temperature: float,
    quantum: bool = False,
    threshold: float = 1,  # in invcm
) -> float:
    if isinstance(frequencies, float):
        frequencies = np.array([frequencies], dtype=float)

    threshold_ = threshold / second * (100 * _c)  # from invcm to ASE
    frequencies = frequencies[np.abs(frequencies) > threshold_]

    # _hplanck in J s
    # _k in J / K
    if quantum:
        arg = (-1.0) * _hplanck * frequencies * second / (_k * temperature)
        F = kB * temperature * np.sum(np.log(1 - np.exp(arg)))
        F += _hplanck * J * second * np.sum(frequencies) / 2
    else:
        constant = kB * temperature * np.log(_hplanck)
        actual = np.log(frequencies / (kB * temperature))
        F = len(frequencies) * constant + kB * temperature * np.sum(actual)
    F /= kB * temperature
    return F


harmonic_free_energy = python_app(_harmonic_free_energy, executors=["default_threads"])


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
    client_args: list[list[str]],
    command_server: str,
    command_client: str,
    stdout: str = "",
    stderr: str = "",
    inputs: list = [],
    outputs: list = [],
    parsl_resource_specification: Optional[dict] = None,
) -> str:
    command_start = make_start_command(command_server, inputs[0], inputs[1])
    commands_client = []
    for i, name in enumerate(hamiltonian_names):
        args = client_args[i]
        assert len(args) == 1  # only have one client per hamiltonian
        for arg in args:
            commands_client += make_client_command(command_client, name, inputs[2 + i], inputs[1], arg),

    command_end = f'{command_server} --cleanup'
    command_copy = f'cp i-pi.output_full.hess {outputs[0]}'

    command_list = [
        TMP_COMMAND,
        CD_COMMAND,
        command_start,
        *commands_client,
        "wait",
        command_end,
        command_copy,
    ]
    return "\n".join(command_list)


execute_ipi = bash_app(_execute_ipi, executors=["ModelEvaluation"])


@typeguard.typechecked
def compute_harmonic(
    state: Union[Geometry, AppFuture],
    hamiltonian: Hamiltonian,
    mode: str = "fd",
    asr: str = "crystal",
    pos_shift: float = 0.01,
    energy_shift: float = 0.00095,
) -> AppFuture:
    hamiltonian: MixtureHamiltonian = 1 * hamiltonian
    names = label_forces(hamiltonian)
    sockets = setup_sockets(names)
    forces = make_force_xml(hamiltonian, names)

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
    inputs += serialize_mixture(hamiltonian, dtype="float64")

    client_args = []
    for name in names:
        args = definition.get_client_args(name, 1, "vibrations")
        client_args.append(args)
    outputs = [
        context.new_file("hess_", ".txt"),
    ]

    command_server = definition.server_command()
    command_client = definition.client_command()
    resources = definition.wq_resources(1)

    result = execute_ipi(
        names,
        client_args,
        command_server,
        command_client,
        stdout=parsl.AUTO_LOGNAME,
        stderr=parsl.AUTO_LOGNAME,
        inputs=inputs,
        outputs=outputs,
        parsl_resource_specification=resources,
    )
    return multiply(load_numpy(inputs=[result.outputs[0]]), Ha / Bohr**2)