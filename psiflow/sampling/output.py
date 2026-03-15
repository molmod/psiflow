import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, InitVar
from typing import Optional, Union

import numpy as np
from ipi.utils.parsing import read_output
from parsl.app.app import python_app
from parsl.app.futures import DataFuture, File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Hamiltonian, MixtureHamiltonian, Zero
from psiflow.sampling.walker import Walker
from psiflow.utils.io import save_npz
from psiflow.utils.parse import get_task_name_id


logger = logging.getLogger(__name__)


DEFAULT_OBSERVABLES = [
    "time{picosecond}",
    "temperature{kelvin}",
    "potential{electronvolt}",
    "volume{angstrom3}",
]


class Status(Enum):
    UNKNOWN = -1
    DONE = 0
    TIMEOUT = 1
    FORCE_EXCEEDED = 2
    OOM = 3
    BROKEN_PIPE = 4
    EXPLODED = 5


@dataclass
class HamiltonianComponent:
    name: str
    hamiltonian: Hamiltonian
    shared: bool

    def __post_init__(self):
        self.address = f"psiflow_{self.name.lower()}"


def split_units(key: str) -> tuple[str, str]:
    name, unit = key.split("{")
    return name, unit[:-1]


def potential_component_name(n: int) -> str:
    return f"pot_component_raw({n}){{electronvolt}}"


def get_simulation_status(stdout: str, stderr: str) -> Status:
    content = Path(stdout).read_text()
    if "@PSIFLOW: We are done here" in content:
        return Status.DONE
    elif "@SOFTEXIT: Kill signal received" in content:
        return Status.TIMEOUT  # i-Pi intercepts SIG_INT and SIG_TERM by default
    elif "@PSIFLOW: Simulation went boom" in content:
        return Status.EXPLODED

    content = Path(stderr).read_text()
    if "force exceeded" in content:
        return Status.FORCE_EXCEEDED
    elif "BrokenPipeError: [Errno 32] Broken pipe" in content:
        return Status.BROKEN_PIPE
    elif "Killed" in content:
        return Status.OOM
    return Status.UNKNOWN


def _parse_simulation_data(
    state: Geometry,
    observables: list[str],
    start: int,
    file_props: File,
    simulation_stdout: str,
    simulation_stderr: str,
) -> tuple[Status, dict, float, float]:
    """"""
    status = get_simulation_status(simulation_stdout, simulation_stderr)
    try:
        # read_output strips unit information from keys
        values, _ = read_output(file_props.filepath)
        data = {k: values[split_units(k)[0]][start:] for k in observables}
        temperature = data["temperature{kelvin}"][-1]
    except IndexError as e:
        # nothing was written
        data = {k: np.array([]) for k in observables}
        temperature = np.nan

    task_id = get_task_name_id(simulation_stdout)[-1]
    logger.info(f"Simulation [ID {task_id}]: {status}")
    return status, data, temperature, state.order.pop("time")


parse_simulation_data = python_app(
    _parse_simulation_data, executors=["default_threads"]
)


def _log_status(
    task_id: str,
    status: Status,
    observables: list[str],
    time: float,
    temperature: float,
    component_map: dict[str, HamiltonianComponent],
    bias_components: list[HamiltonianComponent],
) -> None:
    """Print out some simulation stats"""
    info = [
        f"Simulation [ID {task_id}]",
        f"Status: {status.name}",
        f"Elapsed time: {time*1000:.1f} fs",
        f"Final temperature: {temperature:.1f} K",
        f"Observables: {observables}",
        f"Force components: {[f'{c.name} -> {k}' for k, c in component_map.items()]}",
        f"Bias components: {[c.name for c in bias_components]}",
    ]
    print(*info, sep="\n")


log_status = python_app(_log_status, executors=["default_threads"])


def _add_contributions(
    coefficients: tuple[float, ...],
    *values: np.ndarray,
) -> np.ndarray:
    assert len(coefficients) == len(values) and values[0] is not None
    total = np.zeros(len(values[0]))
    for c, v in zip(coefficients, values):
        total += c * v
    return total


add_contributions = python_app(_add_contributions, executors=["default_threads"])


@psiflow.register_serializable
@dataclass
class SimulationOutput:
    task_id: str
    walker: Walker
    status: Status | AppFuture
    state: Geometry | AppFuture
    data: AppFuture
    time: float | AppFuture
    temperature: float | AppFuture
    trajectory: Optional[Dataset]
    observables: list[str]

    hamiltonian_components: InitVar[list[HamiltonianComponent]]
    component_map: dict[str, HamiltonianComponent] = field(init=False)
    force_comps: list[HamiltonianComponent] = field(init=False)
    bias_comps: list[HamiltonianComponent] = field(init=False)
    _data: dict[str, AppFuture] = field(init=False)

    def __post_init__(self, hamiltonian_components):
        self._data = {}  # to cache deferred_getitem calls
        self.force_comps = [comp for comp in hamiltonian_components if comp.shared]
        self.bias_comps = [comp for comp in hamiltonian_components if not comp.shared]

        # map generic i-Pi keys to hamiltonian force components
        self.component_map = {
            potential_component_name(i): comp for i, comp in enumerate(self.force_comps)
        }

    def __getitem__(self, key: str) -> AppFuture:
        assert key in self.observables
        if key not in self._data:
            self._data[key] = self.data[key]
        return self._data[key]

    def update_walker(self):
        # TODO: when do we want to reset?
        self.walker.state = self.state

    def log_status(self):
        log_status(
            self.task_id,
            self.status,
            self.observables,
            self.time,
            self.temperature,
            self.component_map,
            self.bias_comps,
        )

    def get_energy(self, hamiltonian: Hamiltonian) -> AppFuture:
        """Use stored energy contributions from i-Pi to evaluate a custom (Mixture)Hamiltonian."""
        if hamiltonian == Zero():
            # future because array length not known
            return add_contributions((0.0,), self.data[DEFAULT_OBSERVABLES[0]])

        f_comps = self.force_comps
        f_hamiltonian = MixtureHamiltonian(
            [comp.hamiltonian for comp in f_comps], [1] * len(f_comps)
        )
        coefficients = f_hamiltonian.get_coefficients(1.0 * hamiltonian)
        if coefficients is None:
            raise ValueError(
                f"Provided hamiltonian is not fully in i-Pi forces. "
                f"Make sure you do not ask for bias contributions."
            )
        return add_contributions(
            coefficients, *[self[name] for name in self.component_map]
        )

    def save_data(self, path: Union[str, Path], **kwargs: np.ndarray) -> DataFuture:
        component_map = {c.name: k for k, c in self.component_map.items()}
        future = save_npz(self.data, outputs=[File(path)], **component_map, **kwargs)
        return future.outputs[0]

    @classmethod
    def from_md(
        cls,
        walker: Walker,
        state: AppFuture[Geometry],
        observables: list[str],
        hamiltonian_components: list[HamiltonianComponent],
        start: int,
        file_props: DataFuture,
        file_traj: Optional[DataFuture],
        output_log: str,
        error_log: str,
    ) -> "SimulationOutput":
        """"""
        future = parse_simulation_data(
            state, observables, start, file_props, output_log, error_log
        )
        status, data, temperature, time = future[0], future[1], future[2], future[3]
        trajectory = None
        if file_traj:
            trajectory = Dataset(None, file_traj)
            if start > 0:
                trajectory = trajectory[start:]
        return cls(
            get_task_name_id(output_log)[-1],
            walker,
            status,
            state,
            data,
            time,
            temperature,
            trajectory,
            observables,
            hamiltonian_components,
        )
