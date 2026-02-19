import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Union, Iterable

import parsl
import numpy as np
from parsl.app.app import bash_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.data import Dataset
from psiflow.data.utils import read_frames
from psiflow.hamiltonians import Hamiltonian, MixtureHamiltonian, Zero, MACEHamiltonian
from psiflow.sampling.output import (
    DEFAULT_OBSERVABLES,
    SimulationOutput,
    potential_component_name,
    HamiltonianComponent,
)
from psiflow.sampling.utils import create_xml_list
from psiflow.sampling.walker import Coupling, Walker, partition, Ensemble
from psiflow.utils.io import _save_xml
from psiflow.utils import TMP_COMMAND, CD_COMMAND, export_env_command
from psiflow.sampling.driver import __file__ as PATH_DRIVER


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

    def __eq__(self, other: "EnsembleTable") -> bool:
        if (
            not isinstance(other, EnsembleTable)
            or len(self) != len(other)
            or set(self.keys) != set(other.keys)
            or self.weights.shape != other.weights.shape
        ):
            return False
        key_order = [other.keys.index(key) for key in self.keys]
        return np.all(self.weights == other.weights[:, key_order])


def template(
    walkers: list[Walker],
) -> tuple[list[HamiltonianComponent], EnsembleTable, list[AppFuture]]:
    # multiply by 1.0 to ensure result is Mixture in case len(walkers) == 1
    total_hamiltonian = 1.0 * sum([w.hamiltonian for w in walkers], start=Zero())
    assert not total_hamiltonian == Zero()

    # construct the table of potential / ensemble / bias weights for every system instance
    names = total_hamiltonian.get_named_components()
    weights_h = np.zeros(shape=(len(walkers), len(names)))
    for idx, walker in enumerate(walkers):
        weights_h[idx] = total_hamiltonian.get_coefficients(walker.hamiltonian * 1)

    weights_dict = {}
    if walkers[0].ensemble != Ensemble.NVE:
        weights_dict["TEMP"] = [w.temperature for w in walkers]
    if walkers[0].ensemble in (Ensemble.NPT, Ensemble.NVST):
        weights_dict["PRESSURE"] = [w.pressure for w in walkers]

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
            weights_dict[f"METAD{len(metad_objects) - 1}"] = [
                (i == j) for j in range(len(walkers))
            ]
    plumed_list = [mtd.input() for mtd in metad_objects]

    weights_ensemble = np.array([*weights_dict.values()]).T
    weights_name = tuple(names + list(weights_dict.keys()))
    weights_table = np.concatenate([weights_h, weights_ensemble], axis=-1)

    # make list of all (shared) hamiltonian components
    components = [
        HamiltonianComponent(n, h, all(weights_h[:, idx]))
        for idx, (n, h) in enumerate(zip(names, total_hamiltonian.hamiltonians))
    ]

    return components, EnsembleTable(weights_name, weights_table), plumed_list


def setup_sockets(components: list[HamiltonianComponent]) -> list[ET.Element]:
    xml_str = """
    <ffsocket mode="unix" name="{name}" pbc="False">
        <timeout>{timeout}</timeout>
        <exit_on_disconnect> TRUE </exit_on_disconnect>
        <address>{address}</address>
    </ffsocket>
    """
    timeout = 60 * psiflow.context().definitions["ModelEvaluation"].timeout

    sockets = []
    for comp in components:
        xml = xml_str.format(name=comp.name, address=comp.address, timeout=str(timeout))
        sockets.append(ET.fromstring(xml))
    return sockets


def setup_motion(walker: Walker, fix_com: bool) -> ET.Element:
    if (ensemble := walker.ensemble) == Ensemble.NVE:
        mode = "nve"
    elif ensemble == Ensemble.NVT:
        mode = "nvt"
    elif ensemble in (Ensemble.NPT, Ensemble.NVST):
        mode = "npt"
    else:
        raise ValueError("invalid walker {}".format(walker))
    dynamics = ET.Element("dynamics", mode=mode)
    timestep_element = ET.Element("timestep", units="femtosecond")
    timestep_element.text = str(walker.timestep)
    dynamics.append(timestep_element)

    # thermostat
    tau = ET.Element("tau", units="femtosecond")
    tau.text = "100"  # TODO: hardcoded
    thermostat = ET.Element("thermostat", mode="langevin")
    thermostat.append(tau)
    if ensemble != Ensemble.NVE and walker.pimd:
        thermostat_pimd = ET.Element("thermostat", mode="pile_g")
        thermostat_pimd.append(tau)
        dynamics.append(thermostat_pimd)
    elif ensemble != Ensemble.NVE and not walker.pimd:
        dynamics.append(thermostat)

    # barostat
    if ensemble in (Ensemble.NPT, Ensemble.NVST):
        barostat = ET.Element("barostat", mode="flexible")
        tau = ET.Element("tau", units="femtosecond")
        tau.text = "200"  # TODO: hardcoded
        barostat.append(tau)
        barostat.append(thermostat)  # never use thermostat_pimd here!
        if ensemble == Ensemble.NVST:
            vol_constraint = ET.Element("vol_constraint")
            vol_constraint.text = "True"
            barostat.append(vol_constraint)
        dynamics.append(barostat)

    motion = ET.Element("motion", mode="dynamics")
    motion.append(dynamics)
    fixcom = ET.Element("fixcom")
    fixcom.text = " {} ".format(fix_com)
    motion.append(fixcom)  # ensure kinetic_md ~ temperature
    return motion


def setup_ensemble(
    components: list[HamiltonianComponent], weights_header: tuple[str, ...]
) -> ET.Element:
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


def setup_forces(components: list[HamiltonianComponent]) -> ET.Element:
    forces = ET.Element("forces")
    for comp in components:
        if comp.shared:  # only add components shared across all walkers
            force = ET.Element("force", forcefield=comp.name, weight=comp.name.upper())
            forces.append(force)
    return forces


def setup_ffplumed(nplumed: int) -> list[ET.Element]:
    ffplumed = []
    for i in range(nplumed):
        input_file = ET.Element("file", mode="xyz", cell_units="angstrom")
        input_file.text = "start_0.xyz"  # always present
        plumed_dat = ET.Element("plumed_dat")
        plumed_dat.text = "metad_input{}.txt".format(i)
        ff = ET.Element("ffplumed", name="metad{}".format(i), pbc="False")
        ff.append(input_file)
        ff.append(plumed_dat)
        ffplumed.append(ff)
    return ffplumed


def setup_system_template(
    walkers: list[Walker],
    ensemble_table: EnsembleTable,
    motion: ET.Element,
    ensemble: ET.Element,
    forces: ET.Element,
) -> ET.Element:
    system_template = ET.Element("system_template")
    labels = ET.Element("labels")
    labels.text = create_xml_list(["INDEX"] + [w.upper() for w in ensemble_table.keys])
    system_template.append(labels)

    for i in range(len(ensemble_table)):
        instance = ET.Element("instance")
        instance.text = create_xml_list(
            [f"{i}"] + [str(w) for w in ensemble_table.get_index(i)]
        )
        system_template.append(instance)

    initialize = ET.Element("initialize", nbeads=str(walkers[0].nbeads))
    start = ET.Element("file", mode="ase", cell_units="angstrom")
    start.text = " start_INDEX.xyz "
    initialize.append(start)
    velocities = ET.Element("velocities", mode="thermal", units="kelvin")
    velocities.text = (
        " TEMP " if "TEMP" in ensemble_table.keys else " 300 "
    )  # valid template parameter
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
        DEFAULT_OBSERVABLES
        + [potential_component_name(i) for i in range(n_forces)]
        + observables
    )
    if len(components) != n_forces:  # store bias force components
        full_list.append("ensemble_bias{electronvolt}")
    observables = list(set(full_list))

    if step is None:
        # TODO: this logic should be elsewhere
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


def make_server_command(
    command: str,
    input_xml: File,
    start_xyz: File,
    output_xyz: File,
    output_props: list[File],
    output_trajs: list[File],
) -> str:
    """"""
    args = [
        f"{command}",
        f"--input_xml={input_xml.filepath}",
        f"--start_xyz={start_xyz.filepath}",
        f"--output_xyz={output_xyz.filepath}",
    ]
    if output_props:
        args.append(
            "--output_props=" + ",".join([file.filepath for file in output_props])
        )
    if output_trajs:
        args.append(
            "--output_traj=" + ",".join([file.filepath for file in output_trajs])
        )
    return " ".join(args + ["&"])


def make_driver_commands(
    driver_kwargs: list[dict], file_xyz: File, files_hamiltonian: list[File]
) -> list[str]:
    """"""
    assert len(driver_kwargs) >= len(files_hamiltonian)
    default = f'i-pi-driver-py -u -S "" -m custom -P {PATH_DRIVER} -a {{address}} -o {{options}} &'

    commands = []
    for kwargs in driver_kwargs:
        # drivers know which hamiltonian to load by 'idx'
        address, idx = kwargs.pop("address"), kwargs.pop("idx")
        options = [
            str(file_xyz),
            str(files_hamiltonian[idx]),
            *[f"{k}={v}" for k, v in kwargs.items() if v is not None],
        ]
        commands.append(default.format(address=address, options=",".join(options)))
    return commands


def make_wait_for_sockets_command(
    addresses: Iterable[str], timeout: int = 10, interval: float = 0.1
) -> str:
    """"""
    exist = " && ".join([f"[ -e {a} ]" for a in addresses])
    return f"t=0; until {exist} || (( t++ >= {timeout} )); do sleep {interval}; done"


def _execute_ipi(
    nwalkers: int,
    driver_kwargs: list[dict],
    keep_trajectory: bool,
    coupling_command: Optional[str],
    command_server: str,
    *plumed_list: str,
    env_vars: dict = {},
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
    inputs: list = [],
    outputs: list = [],
    parsl_resource_specification: Optional[dict] = {},
) -> str:
    """"""
    file_xml, file_xyz_in, *files_in = inputs
    file_xyz_out, files_props = outputs[0], outputs[1 : 1 + nwalkers]
    files_traj = outputs[1 + nwalkers : 1 + 2 * nwalkers] if keep_trajectory else []

    write_command_args = [
        f'echo "{plumed_str}" > metad_input{i}.txt'
        for i, plumed_str in enumerate(plumed_list)
    ]
    command_start = make_server_command(
        command_server, file_xml, file_xyz_in, file_xyz_out, files_props, files_traj
    )
    command_wait = make_wait_for_sockets_command(
        set(d["address"] for d in driver_kwargs)
    )
    commands_driver = make_driver_commands(driver_kwargs, file_xyz_in, files_in)

    command_list = [
        TMP_COMMAND,
        CD_COMMAND,
        "\n".join(write_command_args),
        export_env_command(env_vars),
        command_start,
        command_wait,
        *commands_driver,
        "wait",
    ]
    if coupling_command:
        command_list.append(coupling_command)
    return "\n".join(command_list)


execute_ipi = bash_app(_execute_ipi, executors=["ModelEvaluation"])


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
    if motion_defaults is not None:
        raise NotImplementedError

    # generate i-Pi input XML
    hamiltonian_components, ensemble_table, plumed_list = template(walkers)
    coupling = walkers[0].coupling
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
    sockets = setup_sockets(hamiltonian_components)
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

    # app setup and IO
    context = psiflow.context()
    definition = context.definitions["ModelEvaluation"]
    input_file = context.new_file("input_", ".xml")
    _save_xml(simulation, outputs=[input_file])
    inputs = [
        input_file,
        Dataset([w.state for w in walkers]).extxyz,
    ]

    # figure out i-Pi MD driver configuration
    # how many drivers (force evaluators) with which arguments?
    # remove any Harmonic instances because they are not implemented with sockets     -- TODO: why?
    max_nclients = int(sum([w.nbeads for w in walkers]))
    driver_kwargs = []
    for i, comp in enumerate(hamiltonian_components):
        inputs.append(comp.hamiltonian.serialize_function())
        kwargs = {"idx": i, "address": comp.address, "max_force": max_force}
        if isinstance(comp.hamiltonian, MACEHamiltonian):
            kwargs["dtype"] = "float32"  # TODO: should this be configurable?
            for instance_kwargs in definition.get_driver_devices(max_nclients):
                driver_kwargs.append(kwargs | instance_kwargs)
        else:
            driver_kwargs.append(kwargs)

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

    result = execute_ipi(
        len(walkers),
        driver_kwargs,
        keep_trajectory,
        coupling_copy_command,
        definition.server_command(),
        *plumed_list,  # futures
        env_vars=dict(definition.env_vars),
        inputs=inputs,
        outputs=outputs,
        parsl_resource_specification=definition.wq_resources(max_nclients),
    )

    # process MD output
    simulation_outputs = []
    final_states = read_frames(inputs=result.outputs[:1])
    for idx, walker in enumerate(walkers):
        # TODO: order_parameter out of commission
        # if walker.order_parameter is not None:
        #     state = walker.order_parameter.evaluate(state)
        if walker.metadynamics is not None:
            walker.metadynamics.wait_for(result)
        output = SimulationOutput.from_md(
            walker,
            final_states[idx],
            observables,
            hamiltonian_components,  # order is important
            start,
            result.outputs[idx + 1],
            result.outputs[len(walkers) + 1 + idx] if keep_trajectory else None,
            result.stdout,
            result.stderr,
        )
        output.update_walker()
        simulation_outputs.append(output)

    if coupling is not None:
        coupling.update(result)

    return simulation_outputs


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
