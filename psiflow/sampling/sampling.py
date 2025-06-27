from __future__ import annotations  # necessary for type-guarding class methods

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Union, Iterable

import parsl
import typeguard
import numpy as np
from parsl.app.app import bash_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.data import Dataset
from psiflow.hamiltonians import Hamiltonian, MixtureHamiltonian, Zero
from psiflow.sampling.output import (
    DEFAULT_OBSERVABLES,
    SimulationOutput,
    potential_component_names,
)
from psiflow.sampling.walker import Coupling, Walker, partition
from psiflow.utils.io import save_xml
from psiflow.utils import TMP_COMMAND, CD_COMMAND


@dataclass
class HamiltonianComponent:
    name: str
    hamiltonian: Hamiltonian
    shared: bool


@dataclass
class EnsembleTable:
    keys: tuple[str, ...]
    weights: np.ndarray

    def get_index(self, idx: int) -> np.ndarray:
        return self.weights[idx]

    def get_key(self, key: str) -> np.ndarray:
        return self.weights[:, self.keys.index(key)]

    def __len__(self) -> int:
        return self.weights.shape[0]

    def __eq__(self, other: EnsembleTable) -> bool:
        if (
            not isinstance(other, EnsembleTable) or
            len(self) != len(other) or
            set(self.keys) != set(other.keys) or
            self.weights.shape != other.weights.shape
        ):
            return False
        key_order = [other.keys.index(key) for key in self.keys]
        return np.all(self.weights == other.weights[:, key_order])


def create_xml_list(items: list[str]) -> str:
    """Pure helper"""
    return ' [ {} ] '.format(', '.join(items))


@typeguard.typechecked
def template(
    walkers: list[Walker],
) -> tuple[list[HamiltonianComponent], EnsembleTable, list[AppFuture]]:
    # multiply by 1.0 to ensure result is Mixture in case len(walkers) == 1
    total_hamiltonian = 1.0 * sum([w.hamiltonian for w in walkers], start=Zero())
    assert not total_hamiltonian == Zero()

    # create string names for hamiltonians and sort     TODO: why sort?
    # names = label_forces(total_hamiltonian)
    # names, hamiltonians, coefficients = zip(
    #     *sorted(zip(names, total_hamiltonian.hamiltonians, total_hamiltonian.coefficients))
    # )
    # assert MixtureHamiltonian(hamiltonians, coefficients) == total_hamiltonian
    # total_hamiltonian = MixtureHamiltonian(hamiltonians, coefficients)

    # construct the table of potential / ensemble / bias weights for every system instance
    names = total_hamiltonian.get_named_components()
    weights_hamiltonian = np.zeros(shape=(len(walkers), len(names)))
    for idx, walker in enumerate(walkers):
        weights_hamiltonian[idx] = total_hamiltonian.get_coefficients(walker.hamiltonian * 1)

    components = []
    for idx, (name, hamiltonian) in enumerate(zip(names, total_hamiltonian.hamiltonians)):
        weights = weights_hamiltonian[:, idx]
        components.append(HamiltonianComponent(name, hamiltonian, all(weights)))

    weights_dict = {}
    if walkers[0].nvt or walkers[0].npt:
        weights_dict['TEMP'] = [w.temperature for w in walkers]
    if walkers[0].npt:
        weights_dict['PRESSURE'] = [w.pressure for w in walkers]

    # inspect metadynamics attributes and allocate additional weights per MTD
    metad_objects = []
    for i, walker in enumerate(walkers):
        mtd = walker.metadynamics
        if mtd is not None:
            assert mtd not in metad_objects, (
                "Metadynamics biases need to be "
                "independent for uncoupled walkers. To perform actual "
                "multiple walker metadynamics, use "
                "psiflow.sampling.multiple_walker_metadynamics"
            )
            metad_objects.append(mtd)
            weights_dict[f'METAD{len(metad_objects) - 1}'] = [(i == j) for j in range(len(walkers))]
    plumed_list = [mtd.input() for mtd in metad_objects]

    weights_ensemble = np.array([*weights_dict.values()]).T
    weights_name = tuple(names + list(weights_dict.keys()))
    weights_table = np.concatenate([weights_hamiltonian, weights_ensemble], axis=-1)

    return components, EnsembleTable(weights_name, weights_table), plumed_list


def make_force_xml(hamiltonian: MixtureHamiltonian, names: list[str]) -> ET.Element:
    # TODO: move to relevant module
    forces = ET.Element("forces")
    for n, c in zip(names, hamiltonian.coefficients):
        forces.append(ET.Element("force", forcefield=n, weight=str(c)))
    return forces


@typeguard.typechecked
def setup_sockets(
    hamiltonian_labels: Iterable[str],
) -> list[ET.Element]:
    sockets = []
    for name in hamiltonian_labels:
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
def setup_motion(walker: Walker, fix_com: bool) -> ET.Element:
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
        mode = "npt"
        dynamics = ET.Element("dynamics", mode=mode)
        dynamics.append(timestep_element)
        if walker.pimd:
            dynamics.append(thermostat_pimd)
        else:
            dynamics.append(thermostat)
        mode = "flexible"
        barostat = ET.Element("barostat", mode=mode)
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
    fixcom.text = " {} ".format(fix_com)
    motion.append(fixcom)  # ensure kinetic_md ~ temperature
    return motion


@typeguard.typechecked
def setup_ensemble(components: list[HamiltonianComponent], weights_header: tuple[str, ...]) -> ET.Element:
    ensemble = ET.Element("ensemble")

    # set TEMP in any case to avoid i-PI from throwing weird errors
    temperature = ET.Element("temperature", units="kelvin")
    temperature.text = "TEMP" if "TEMP" in weights_header else " 300 "
    ensemble.append(temperature)
    if "PRESSURE" in weights_header:
        pressure = ET.Element("pressure", units="megapascal")
        pressure.text = "PRESSURE"
        ensemble.append(pressure)

    bias = ET.Element("bias")
    bias_weights_list = []

    # add hamiltonian components that are not shared between all walkers as bias
    # TODO: do we always want to do this?
    for comp in components:
        if not comp.shared:
            force = ET.Element("force", forcefield=comp.name)
            bias.append(force)
            bias_weights_list.append(comp.name.upper())

    # add metadynamics bias if present
    for idx in range(len(weights_header) - len(components)):
        if (name := f"METAD{idx}") in weights_header:
            force = ET.Element("force", forcefield=name.lower())
            bias.append(force)
            bias_weights_list.append(name)

    if bias_weights_list:
        ensemble.append(bias)
        bias_weights = ET.Element("bias_weights")
        bias_weights.text = create_xml_list(bias_weights_list)
        ensemble.append(bias_weights)

    return ensemble


@typeguard.typechecked
def setup_forces(hamiltonian_components: list[HamiltonianComponent]) -> ET.Element:
    forces = ET.Element("forces")
    for comp in hamiltonian_components:
        if comp.shared:                     # only add components shared across all walkers
            force = ET.Element("force", forcefield=comp.name, weight=comp.name.upper())
            forces.append(force)
    return forces


@typeguard.typechecked
def setup_ffplumed(nplumed: int) -> list[ET.Element]:
    ffplumed = []
    for i in range(nplumed):
        input_file = ET.Element("file", mode="xyz", cell_units="angstrom")
        input_file.text = "start_0.xyz"  # always present
        plumeddat = ET.Element("plumeddat")
        plumeddat.text = "metad_input{}.txt".format(i)
        ff = ET.Element("ffplumed", name="metad{}".format(i), pbc="False")
        ff.append(input_file)
        ff.append(plumeddat)
        ffplumed.append(ff)
    return ffplumed


@typeguard.typechecked
def setup_system_template(
    walkers: list[Walker],
    ensemble_table: EnsembleTable,
    motion: ET.Element,
    ensemble: ET.Element,
    forces: ET.Element,
) -> ET.Element:
    system_template = ET.Element("system_template")
    labels = ET.Element("labels")
    labels.text = create_xml_list(['INDEX'] + [w.upper() for w in ensemble_table.keys])
    system_template.append(labels)

    for i in range(len(ensemble_table)):
        print(i)
        print(ensemble_table.get_index(i))
        instance = ET.Element("instance")
        instance.text = create_xml_list([f'{i}'] + [str(w) for w in ensemble_table.get_index(i)])
        system_template.append(instance)

    initialize = ET.Element("initialize", nbeads=str(walkers[0].nbeads))
    start = ET.Element("file", mode="ase", cell_units="angstrom")
    start.text = " start_INDEX.xyz "
    initialize.append(start)
    velocities = ET.Element("velocities", mode="thermal", units="kelvin")
    velocities.text = " TEMP " if "TEMP" in ensemble_table.keys else " 300 "        # valid template parameter
    initialize.append(velocities)
    if walkers[0].masses is not None:
        import ase.units
        AMU_TO_AU = ase.units._amu / ase.units._me
        masses = ET.Element("masses", mode="manual")
        masses.text = create_xml_list([str(i) for i in walkers[0].masses * AMU_TO_AU])
        initialize.append(masses)

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
    components: list[HamiltonianComponent],
    observables: Optional[list[str]],
    step: Optional[int],
    keep_trajectory: bool,
    checkpoint_step: int,
) -> tuple[ET.Element, list]:

    if observables is None:
        observables = []

    n_forces = sum([comp.shared for comp in components], 0)
    full_list = (
        DEFAULT_OBSERVABLES + potential_component_names(n_forces) + observables
    )
    if len(components) != n_forces:             # store bias force components
        full_list.append('ensemble_bias{electronvolt}')
    observables = list(set(full_list))

    if step is None:
        step = checkpoint_step

    output = ET.Element("output", prefix="output")
    checkpoint = ET.Element(
        "checkpoint",
        filename="checkpoint",
        stride=str(checkpoint_step),
        overwrite="True",
    )
    output.append(checkpoint)
    if keep_trajectory:
        trajectory = ET.Element(
            "trajectory",
            filename="trajectory",
            stride=str(step),
            format="ase",
            bead="0",
        )
        trajectory.text = r" positions "
        output.append(trajectory)
    properties = ET.Element(
        "properties",
        filename="properties",
        stride=str(step),
    )
    properties.text = create_xml_list(observables)
    output.append(properties)
    return output, observables


@typeguard.typechecked
def setup_smotion(
    coupling: Optional[Coupling], plumed_list: list[AppFuture]
) -> ET.Element:
    has_metad = len(plumed_list) > 0
    has_coupling = coupling is not None
    if has_coupling:
        smotion = coupling.get_smotion(has_metad)
    else:
        smotion = ET.Element("smotion", mode="dummy")
    if has_metad:
        metaff = ET.Element("metaff")
        bias_names = ["metad{}".format(i) for i in range(len(plumed_list))]
        metaff.text = " [ " + ", ".join(bias_names) + " ] "
        metad = ET.Element("metad")
        metad.append(metaff)
        smotion_metad = ET.Element("smotion", mode="metad")
        smotion_metad.append(metad)
        if has_coupling:
            smotion.append(smotion_metad)
        else:  # overwrite dummy smotion
            smotion = smotion_metad
    return smotion


def make_start_command(command: str, input_xml: File, start_xyz: File, nwalkers: int = 1) -> str:
    """"""
    return f'{command} --nwalkers={nwalkers} --input_xml={input_xml.filepath} --start_xyz={start_xyz.filepath} &'


def make_client_command(command: str, address: str, hamiltonian: File,
                        start: File, arg: str, max_force: float = None) -> str:
    """"""
    return '{c} --address={a} --path_hamiltonian={p} --start={s} {m} {arg} &'.format(
        c=command, a=address.lower(), p=hamiltonian.filepath, s=start.filepath, arg=arg,
        m=(f'--max_force={max_force}' if max_force is not None else ''),
    )


def _execute_ipi(
    nwalkers: int,
    hamiltonian_names: list[str],
    client_args: list[str],
    keep_trajectory: bool,
    max_force: Optional[float],
    coupling_command: Optional[str],
    command_server: str,
    command_client: str,
    *plumed_list: str,
    env_vars: dict = {},
    stdout: str = "",
    stderr: str = "",
    inputs: list = [],
    outputs: list = [],
    parsl_resource_specification: Optional[dict] = {},
) -> str:
    write_command = '\n'.join([
        f'echo "{plumed_str}" > metad_input{i}.txt' for i, plumed_str in enumerate(plumed_list)
    ])
    env_command = 'export ' + ' '.join([f"{name}={value}" for name, value in env_vars.items()])
    command_start = make_start_command(command_server, inputs[0], inputs[1], nwalkers)
    commands_client = []
    for i, name in enumerate(hamiltonian_names):
        args = client_args[i]
        for arg in args:
            commands_client += make_client_command(command_client, name, inputs[2 + i], inputs[1], arg, max_force),

    command_end = f'{command_server} --cleanup --output_xyz={outputs[0].filepath}'
    commands_copy = []
    for i in range(nwalkers):
        commands_copy += f'cp walker-{i}_output.properties {outputs[i + 1].filepath}',
        if keep_trajectory:
            commands_copy += f'cp walker-{i}_output.trajectory_0.extxyz {outputs[i + nwalkers + 1].filepath}',
    if coupling_command is not None:
        commands_copy += coupling_command,
    command_copy = '; '.join(commands_copy)

    command_list = [
        TMP_COMMAND,
        CD_COMMAND,
        write_command,
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
def _sample(
    walkers: list[Walker],
    steps: int,
    step: Optional[int] = None,
    start: int = 0,
    keep_trajectory: bool = True,
    max_force: Optional[float] = None,
    observables: Optional[list[str]] = None,
    motion_defaults: Union[None, str, ET.Element] = None,
    fix_com: bool = True,
    prng_seed: int = 12345,
    checkpoint_step: Optional[int] = None,
    verbosity: str = "medium",
) -> list[SimulationOutput]:
    assert len(walkers) > 0
    hamiltonian_components, ensemble_table, plumed_list = template(walkers)
    coupling = walkers[0].coupling

    if motion_defaults is not None:
        raise NotImplementedError

    motion = setup_motion(walkers[0], fix_com)
    ensemble = setup_ensemble(hamiltonian_components, ensemble_table.keys)
    forces = setup_forces(hamiltonian_components)
    system_template = setup_system_template(
        walkers,
        ensemble_table,
        motion,
        ensemble,
        forces,
    )
    smotion = setup_smotion(coupling, plumed_list)

    # make sure at least one checkpoint is being written
    if checkpoint_step is None:  # default to every 5% of simulation progress
        if step is None:
            checkpoint_step = math.ceil(steps / 20)
        else:
            checkpoint_step = step
    else:
        if steps < checkpoint_step:  # technically a user error
            checkpoint_step = steps
    if step is not None:
        start = math.floor(start / step)  # start is applied on subsampled quantities
    if step is None:
        keep_trajectory = False
    # TODO: check whether observables are valid?
    output, observables = setup_output(
        hamiltonian_components,  # for potential components
        observables,
        step,
        keep_trajectory,
        checkpoint_step,
    )
    simulation = ET.Element(
        "simulation",
        verbosity=str(verbosity),
        safe_stride=str(checkpoint_step),
    )
    sockets = setup_sockets([comp.name for comp in hamiltonian_components])
    for socket in sockets:
        simulation.append(socket)
    ffplumed = setup_ffplumed(len(plumed_list))
    for ff in ffplumed:
        simulation.append(ff)
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

    context = psiflow.context()
    definition = context.definitions["ModelEvaluation"]
    input_future = save_xml(
        simulation,
        outputs=[context.new_file("input_", ".xml")],
    ).outputs[0]
    inputs = [
        input_future,
        Dataset([w.state for w in walkers]).extxyz,
    ]

    # remove any Harmonic instances because they are not implemented with sockets       -- TODO: why?
    max_nclients = int(sum([w.nbeads for w in walkers]))
    client_args = []
    for comp in hamiltonian_components:
        inputs.append(comp.hamiltonian.serialize_function())
        args = definition.get_client_args(comp.name, max_nclients, motion="dynamics")
        client_args.append(args)
    outputs = [context.new_file("data_", ".xyz")]
    outputs += [context.new_file("simulation_", ".txt") for _ in walkers]
    if keep_trajectory:
        outputs += [context.new_file("data_", ".xyz") for _ in walkers]
        assert len(outputs) == 2 * len(walkers) + 1
    else:
        assert len(outputs) == len(walkers) + 1

    # add coupling inputs after all other ones;
    # these are updated again with the corresponding outputs from execute_ipi
    # the if/else stuff has to happen outside of bash_app because coupling cannot
    # be passed into a bash app as it cannot be serialized
    if coupling is not None:
        inputs += coupling.inputs()
        outputs += [File(f.filepath) for f in coupling.inputs()]
        coupling_copy_command = coupling.copy_command()
    else:
        coupling_copy_command = None

    command_server = definition.server_command()
    command_client = definition.client_command()
    resources = definition.wq_resources(max_nclients)
    result = execute_ipi(
        len(walkers),
        [comp.name for comp in hamiltonian_components],
        client_args,
        keep_trajectory,
        max_force,
        coupling_copy_command,
        command_server,
        command_client,
        *plumed_list,
        env_vars=dict(definition.env_vars),
        stdout=parsl.AUTO_LOGNAME,
        stderr=parsl.AUTO_LOGNAME,
        inputs=inputs,
        outputs=outputs,
        parsl_resource_specification=resources,
    )

    final_states = Dataset(None, result.outputs[0])
    simulation_outputs = [SimulationOutput(observables) for _ in range(len(walkers))]
    for i, simulation_output in enumerate(simulation_outputs):
        state = final_states[i]
        if walkers[i].order_parameter is not None:
            state = walkers[i].order_parameter.evaluate(state)
        simulation_output.parse(result, state)
        simulation_output.parse_data(
            start,
            result.outputs[i + 1],
            hamiltonians=[comp.hamiltonian for comp in hamiltonian_components],
        )
        if keep_trajectory:
            j = len(walkers) + 1 + i
            trajectory = Dataset(None, result.outputs[j])
            if start > 0:
                trajectory = trajectory[start:]
            simulation_output.trajectory = trajectory
        if walkers[i].metadynamics is not None:
            walkers[i].metadynamics.wait_for(result)
        simulation_output.update_walker(walkers[i])

    if coupling is not None:
        coupling.update(result)

    return simulation_outputs


@typeguard.typechecked
def sample(
    walkers: list[Walker],
    steps: int,
    step: Optional[int] = None,
    start: int = 0,
    keep_trajectory: bool = True,
    max_force: Optional[float] = None,
    observables: Optional[list[str]] = None,
    motion_defaults: Union[None, str, ET.Element] = None,
    fix_com: bool = True,
    prng_seed: int = 12345,
    use_unique_seeds: bool = True,
    checkpoint_step: Optional[int] = None,
    verbosity: str = "medium",
) -> list[SimulationOutput]:
    indices = partition(walkers)
    outputs = [None] * len(walkers)
    for i, group in enumerate(indices):
        if not use_unique_seeds:
            seed = prng_seed
        else:
            seed = prng_seed + i
        _walkers = [walkers[index] for index in group]
        _outputs = _sample(
            _walkers,
            steps,
            step=step,
            start=start,
            keep_trajectory=keep_trajectory,
            max_force=max_force,
            observables=observables,
            motion_defaults=motion_defaults,
            fix_com=fix_com,
            prng_seed=seed,
            checkpoint_step=checkpoint_step,
            verbosity=verbosity,
        )
        for i, index in enumerate(group):
            outputs[index] = _outputs[i]
    return outputs
