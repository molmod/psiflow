from __future__ import annotations  # necessary for type-guarding class methods

import xml.etree.ElementTree as ET
from typing import Optional, Union

import parsl
import typeguard
from parsl.app.app import bash_app

import psiflow
from psiflow.data import Dataset
from psiflow.hamiltonians.hamiltonian import Hamiltonian, MixtureHamiltonian, Zero
from psiflow.sampling.coupling import Coupling
from psiflow.sampling.output import SimulationOutput
from psiflow.sampling.walker import Walker, partition
from psiflow.utils import save_xml


@typeguard.typechecked
def template(walkers: list[Walker]) -> tuple[dict[str, Hamiltonian], list[tuple]]:
    assert len(partition(walkers)) == 1
    # multiply by 1.0 to ensure result is Mixture in case len(walkers) == 1
    total_hamiltonian = 1.0 * sum([w.hamiltonian for w in walkers], start=Zero())

    # create string names for hamiltonians and sort
    names = []
    counts = {}
    for h in total_hamiltonian.hamiltonians:
        if h.__class__.__name__ not in counts:
            counts[h.__class__.__name__] = 0
        count = counts.get(h.__class__.__name__)
        counts[h.__class__.__name__] += 1
        names.append(h.__class__.__name__ + str(count))
    _, hamiltonians = zip(*sorted(zip(names, total_hamiltonian.hamiltonians)))
    _, coefficients = zip(*sorted(zip(names, total_hamiltonian.coefficients)))
    hamiltonians = list(hamiltonians)
    coefficients = list(coefficients)
    names = sorted(names)
    assert MixtureHamiltonian(hamiltonians, coefficients) == total_hamiltonian
    total_hamiltonian = MixtureHamiltonian(hamiltonians, coefficients)

    weights_header = tuple(names)
    if walkers[0].npt:
        weights_header = ("TEMP", "PRESSURE") + weights_header
    elif walkers[0].nvt:
        weights_header = ("TEMP",) + weights_header
    else:
        pass

    weights_table = [weights_header]
    for walker in walkers:
        coefficients = total_hamiltonian.get_coefficients(1.0 * walker.hamiltonian)
        if walker.npt:
            ensemble = (walker.temperature, walker.pressure)
        elif walker.nvt:
            ensemble = (walker.temperature,)
        else:
            ensemble = ()
        weights_table.append(ensemble + tuple(coefficients))

    hamiltonians_map = {n: h for n, h in zip(names, hamiltonians)}
    return hamiltonians_map, weights_table


@typeguard.typechecked
def setup_motion(walker: Walker) -> ET.Element:
    timestep_element = ET.Element("timestep", units="femtosecond")
    timestep_element.text = str(walker.timestep)

    tau = ET.Element("tau", units="femtosecond")
    tau.text = "100"
    thermostat_pimd = ET.Element("thermostat", mode="pile_g")
    thermostat_pimd.append(tau)
    thermostat = ET.Element("thermostat", mode="langevin")
    thermostat.append(tau)
    if walker.nve:
        dynamics = ET.Element("dynamics", mode="nve")
        dynamics.append(timestep_element)
    elif walker.nvt:
        dynamics = ET.Element("dynamics", mode="nvt")
        dynamics.append(timestep_element)
        if walker.pimd:
            dynamics.append(thermostat_pimd)
        else:
            dynamics.append(thermostat)
    elif walker.npt:
        dynamics = ET.Element("dynamics", mode="npt")
        dynamics.append(timestep_element)
        if walker.pimd:
            dynamics.append(thermostat_pimd)
        else:
            dynamics.append(thermostat)
        barostat = ET.Element("barostat", mode="flexible")
        tau = ET.Element("tau", units="femtosecond")
        tau.text = "200"
        barostat.append(tau)
        barostat.append(thermostat)  # never use thermostat_pimd here!
        dynamics.append(barostat)
    else:
        raise ValueError("invalid walker {}".format(walker))

    motion = ET.Element("motion", mode="dynamics")
    motion.append(dynamics)
    fixcom = ET.Element("fixcom")
    fixcom.text = " False "
    motion.append(fixcom)  # ensure kinetic_md ~ temperature
    return motion


@typeguard.typechecked
def setup_ensemble(weights_header: tuple[str, ...]) -> ET.Element:
    ensemble = ET.Element("ensemble")
    if "TEMP" in weights_header:
        temperature = ET.Element("temperature", units="kelvin")
        temperature.text = "TEMP"
        ensemble.append(temperature)
    if "PRESSURE" in weights_header:
        pressure = ET.Element("pressure", units="megapascal")
        pressure.text = "PRESSURE"
        ensemble.append(pressure)
    return ensemble


@typeguard.typechecked
def setup_forces(weights_header: tuple[str, ...]) -> ET.Element:
    forces = ET.Element("forces")
    for name in weights_header:
        if name in ["TEMP", "PRESSURE"]:
            continue
        force = ET.Element("force", forcefield=name, weight=name.upper())
        forces.append(force)
    return forces


@typeguard.typechecked
def setup_sockets(
    hamiltonians_map: dict[str, Hamiltonian],
) -> list[ET.Element]:
    sockets = []
    for name in hamiltonians_map.keys():
        ffsocket = ET.Element("ffsocket", mode="unix", name=name, pbc="False")
        timeout = ET.Element("timeout")
        timeout.text = str(60 * psiflow.context()["ModelEvaluation"].timeout)
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
def setup_system_template(
    walkers: list[Walker],
    weights_table: list[tuple],
    motion: ET.Element,
    ensemble: ET.Element,
    forces: ET.Element,
) -> ET.Element:
    system_template = ET.Element("system_template")
    labels = ET.Element("labels")
    _labels = [w.upper() for w in weights_table[0]]
    labels.text = "[ INDEX, {} ]".format(", ".join(_labels))
    system_template.append(labels)

    for i, weights in enumerate(weights_table[1:]):
        instance = ET.Element("instance")
        instance.text = "[ {}, {}]".format(i, ", ".join([str(w) for w in weights]))
        system_template.append(instance)

    initialize = ET.Element("initialize", nbeads=str(walkers[0].nbeads))
    start = ET.Element("file", mode="ase", cell_units="angstrom")
    start.text = " start_INDEX.xyz "
    initialize.append(start)
    velocities = ET.Element("velocities", mode="thermal", units="kelvin")
    velocities.text = " TEMP "
    initialize.append(velocities)

    system = ET.Element("system", prefix="walker-INDEX")
    system.append(initialize)
    system.append(motion)
    system.append(ensemble)
    system.append(forces)

    template = ET.Element("template")
    template.append(system)
    system_template.append(template)
    return system_template


@typeguard.typechecked
def setup_output(
    nwalkers: int,
    observables: list[str],
    step: Optional[int],
    checkpoint_step: int,
) -> tuple[ET.Element, list]:
    output = ET.Element("output", prefix="output")
    if step is not None:
        checkpoint_step = step
        trajectory = ET.Element(
            "trajectory",
            filename="trajectory",
            stride=str(step),
            format="ase",
        )
        trajectory.text = r" positions{angstrom} "
        output.append(trajectory)
    checkpoint = ET.Element(
        "checkpoint",
        filename="checkpoint",
        stride=str(checkpoint_step),
        overwrite="True",
    )
    output.append(checkpoint)
    properties = ET.Element(
        "properties",
        filename="properties",
        stride=str(checkpoint_step),
    )
    properties.text = " [ " + ", ".join(observables) + " ] "
    output.append(properties)

    # TODO: check whether observables are valid
    simulation_outputs = [SimulationOutput(observables) for i in range(nwalkers)]
    return output, simulation_outputs


def _execute_ipi(
    nwalkers: int,
    hamiltonian_names: list[str],
    client_args: list[str],
    keep_trajectory: bool,
    max_force: Optional[float],
    command_server: str,
    command_client: str,
    stdout: str = "",
    stderr: str = "",
    inputs: list = [],
    outputs: list = [],
) -> str:
    tmp_command = 'tmpdir=$(mktemp -d -t "ipi_XXXXXXXXX");'
    cd_command = "cd $tmpdir;"
    command_start = command_server + " --nwalkers={}".format(nwalkers)
    command_start += " --input_xml={}".format(inputs[0].filepath)
    command_start += " --start_xyz={}".format(inputs[1].filepath)
    command_start += "  & \n"
    command_clients = ""
    for i, name in enumerate(hamiltonian_names):
        address = name.lower()
        command_ = command_client + " --address={}".format(address)
        command_ += " --path_hamiltonian={}".format(inputs[2 + i].filepath)
        command_ += " --start={}".format(inputs[1].filepath)
        if max_force is not None:
            command_ += " --max_force={}".format(max_force)
        command_ += " " + client_args[i] + " "
        command_ += " & \n"
        command_clients += command_

    # command_pid = 'pid=$!; '
    # command_wait = 'wait $pid; '
    command_end = command_server
    command_end += " --cleanup"
    command_end += " --output_xyz={};".format(outputs[0].filepath)
    command_copy = ""
    for i in range(nwalkers):
        command_copy += "cp walker-{}_output.properties {}; ".format(
            i,
            outputs[i + 1].filepath,
        )
    if keep_trajectory:
        for i in range(nwalkers):
            command_copy += "cp walker-{}_output.trajectory_0.ase {}; ".format(
                i,
                outputs[i + nwalkers + 1].filepath,
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
def sample(
    walkers: list[Walker],
    steps: int,
    step: Optional[int] = None,
    max_force: Optional[float] = None,
    observables: Optional[list[str]] = None,
    coupling: Optional[Coupling] = None,
    motion_defaults: Union[None, str, ET.Element] = None,
    prng_seed: int = 12345,
    checkpoint_step: int = 100,
) -> list[SimulationOutput]:
    assert len(walkers) > 0
    hamiltonians_map, weights_table = template(walkers)

    if motion_defaults is not None:
        raise NotImplementedError

    motion = setup_motion(walkers[0])
    ensemble = setup_ensemble(weights_table[0])
    forces = setup_forces(weights_table[0])
    system_template = setup_system_template(
        walkers,
        weights_table,
        motion,
        ensemble,
        forces,
    )

    sockets = setup_sockets(hamiltonians_map)
    if observables is None:
        observables = [
            "time{picosecond}",
            "temperature{kelvin}",
            "potential{electronvolt}",
        ]
    output, simulation_outputs = setup_output(
        len(walkers),
        observables,
        step,
        checkpoint_step,
    )

    smotion = ET.Element("smotion", mode="dummy")

    simulation = ET.Element("simulation", verbosity="high")
    for socket in sockets:
        simulation.append(socket)
    simulation.append(output)
    simulation.append(system_template)
    simulation.append(smotion)

    total_steps = ET.Element("total_steps")
    total_steps.text = " {} ".format(steps)
    simulation.append(total_steps)

    prng = ET.Element("prng")
    seed = ET.Element("seed")
    seed.text = " {} ".format(prng_seed)
    prng.append(seed)
    simulation.append(prng)

    # execute with i-PI
    context = psiflow.context()
    input_future = save_xml(
        simulation,
        outputs=[context.new_file("input_", ".xml")],
    ).outputs[0]
    inputs = [
        input_future,
        Dataset([w.state for w in walkers]).data_future,
    ]
    inputs += [h.serialize() for h in hamiltonians_map.values()]
    hamiltonian_names = list(hamiltonians_map.keys())
    client_args = []
    for name in hamiltonian_names:
        nclients, args = context["ModelEvaluation"].get_client_args(name, len(walkers))
        client_args.append(args)
    outputs = [context.new_file("data_", ".xyz")]
    outputs += [context.new_file("simulation_", ".txt") for w in walkers]
    if step is not None:
        outputs += [context.new_file("data_", ".xyz") for w in walkers]
        assert len(outputs) == 2 * len(walkers) + 1
    else:
        assert len(outputs) == len(walkers) + 1

    result = execute_ipi(
        len(walkers),
        hamiltonian_names,
        client_args,
        (step is not None),
        max_force=max_force,
        command_server=context["ModelEvaluation"].server_command(),
        command_client=context["ModelEvaluation"].client_command(),
        stdout=parsl.AUTO_LOGNAME,
        stderr=parsl.AUTO_LOGNAME,
        inputs=inputs,
        outputs=outputs,
    )

    final_states = Dataset(None, data_future=result.outputs[0])

    for i, simulation_output in enumerate(simulation_outputs):
        simulation_output.parse(result, final_states[i])
        simulation_output.parse_data(result.outputs[i + 1])
        if step is not None:
            j = len(walkers) + 1 + i
            trajectory = Dataset(None, data_future=result.outputs[j])
            simulation_output.trajectory = trajectory
        walkers[i].update(simulation_output)
    return simulation_outputs
