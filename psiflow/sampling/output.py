import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry, NullState
from psiflow.hamiltonians.hamiltonian import Hamiltonian, MixtureHamiltonian
from psiflow.utils import unpack_i

DEFAULT_OBSERVABLES = [
    "time{picosecond}",
    "temperature{kelvin}",
    "potential{electronvolt}",
]


@typeguard.typechecked
def potential_component_names(n: int):
    str_format = "pot_component_raw({})"
    return [str_format.format(i) + "{electronvolt}" for i in range(n)]


@typeguard.typechecked
def read_output(filename):  # from i-PI
    # Regex pattern to match header lines and capture relevant parts
    header_pattern = re.compile(
        r"#\s*(column|cols\.)\s+(\d+)(?:-(\d+))?\s*-->\s*([^\s\{]+)(?:\{([^\}]+)\})?\s*:\s*(.*)"
    )

    # Reading the file
    with open(filename, "r") as file:
        lines = file.readlines()

    header_lines = [line for line in lines if line.startswith("#")]
    data_lines = [line for line in lines if not line.startswith("#") and line.strip()]

    # Interprets properties
    properties = {}
    for line in header_lines:
        match = header_pattern.match(line)
        if match:
            # Extracting matched groups
            (
                col_type,
                start_col,
                end_col,
                property_name,
                units,
                description,
            ) = match.groups()
            col_info = f"{start_col}-{end_col}" if end_col else start_col
            properties[col_info] = {
                "name": property_name,
                "units": units,
                "description": description,
            }

    # Parse data
    values_dict = {}
    info_dict = {}
    for prop_info in properties.values():
        # Initialize list to hold values for each property
        values_dict[prop_info["name"]] = []
        # Save units and description
        info_dict[prop_info["name"]] = (prop_info["units"], prop_info["description"])

    for line in data_lines:
        values = line.split()
        for column_info, prop_info in properties.items():
            if "-" in column_info:  # Multi-column property
                start_col, end_col = map(
                    int, column_info.split("-")
                )  # 1-based indexing
                prop_values = values[
                    start_col - 1 : end_col
                ]  # Adjust to 0-based indexing
            else:  # Single column property
                col_index = int(column_info) - 1  # Adjust to 0-based indexing
                prop_values = [values[col_index]]

            values_dict[prop_info["name"]].append([float(val) for val in prop_values])

    for prop_name, prop_values in values_dict.items():
        values_dict[prop_name] = np.array(
            prop_values
        ).squeeze()  # make 1-col into a flat array

    return values_dict, info_dict


@typeguard.typechecked
def _parse_data(
    status: int,
    keys: list[str],
    inputs: list = [],
) -> list[np.ndarray]:
    from psiflow.sampling.output import read_output

    bare_keys = []
    for key in keys:
        if "{" in key:
            bare_key = key.split("{")[0]
        else:
            bare_key = key
        bare_keys.append(bare_key)
    if status not in [0, 1]:
        return [np.zeros(0) for key in bare_keys]  # read_output might fail
    values, _ = read_output(inputs[0].filepath)
    return [values[key] for key in bare_keys]


parse_data = python_app(_parse_data, executors=["default_threads"])


@typeguard.typechecked
def _add_contributions(
    coefficients: tuple[float, ...],
    *values: np.ndarray,
) -> np.ndarray:
    assert len(coefficients) == len(values)
    total = np.zeros(len(values[0]))
    for i, c in enumerate(coefficients):
        total += c * values[i]
    return total


add_contributions = python_app(_add_contributions, executors=["default_threads"])


@typeguard.typechecked
def _parse(
    state: Geometry,
    inputs: list = [],
) -> tuple[float, float, int]:
    time = state.order.pop("time")
    temperature = state.order.pop("temperature")

    # determine status based on stdout
    with open(inputs[0], "r") as f:
        content = f.read()
    if "force exceeded" in content:
        status = 2  # max_force exception
    elif "@SOFTEXIT: Kill signal received" in content:
        status = 1  # timeout
    elif "@ SIMULATION: Exiting cleanly" in content:
        status = 0  # everything OK
    else:
        status = -1
    return time, temperature, status


parse = python_app(_parse, executors=["default_threads"])


def _update_walker(
    status: int,
    state: Geometry,
    start: Geometry,
) -> Geometry:
    from psiflow.geometry import NullState

    if (status in [0, 1]) and state != NullState:
        return state
    else:
        return start


update_walker = python_app(_update_walker, executors=["default_threads"])


@typeguard.typechecked
def _get_state(
    geometry: Geometry,
    status: int,
) -> Geometry:
    if status in [0, 1]:
        return geometry
    else:
        return NullState


get_state = python_app(_get_state, executors=["default_threads"])


@typeguard.typechecked
@psiflow.serializable
class SimulationOutput:
    """Gathers simulation output

    status is an integer which represents an exit code of the run:

    -1: unknown error
     0: run completed successfully
     1: run terminated early due to time limit
     2: run terminated early due to max force exception

    """

    _data: dict[str, Optional[AppFuture]]
    state: Union[Geometry, AppFuture, None]
    stdout: Optional[str]
    status: Union[int, AppFuture, None]
    time: Union[float, AppFuture, None]
    temperature: Union[float, AppFuture, None]
    trajectory: Optional[Dataset]

    def __init__(self, fields: list[str]):
        self._data = {key: None for key in fields}

        self.state = None
        self.stdout = None
        self.status = None
        self.time = None
        self.temperature = None
        self.trajectory = None
        self.hamiltonians = None

    def __getitem__(self, key: str) -> AppFuture:
        if self._data.get(key, None) is None:
            raise ValueError("output {} not available".format(key))
        return self._data[key]

    def parse(
        self,
        result: AppFuture,  # result from ipi execution
        state: AppFuture,
    ):
        self.state = state
        self.stdout = result.stdout
        parsed = parse(state, inputs=[result.stdout, result.stderr])
        self.time = unpack_i(parsed, 0)
        self.temperature = unpack_i(parsed, 1)
        self.status = unpack_i(parsed, 2)

    def parse_data(
        self,
        data_future: DataFuture,
        hamiltonians: list[Hamiltonian],
    ):
        data = parse_data(
            self.status,
            list(self._data.keys()),
            inputs=[data_future],
        )
        for i, key in enumerate(self._data.keys()):
            self._data[key] = unpack_i(data, i)
        self.hamiltonians = hamiltonians

    def update_walker(self, walker):
        walker.state = update_walker(
            self.status,
            self.state,
            walker.start,
        )

    def get_energy(self, hamiltonian: Hamiltonian) -> AppFuture:
        all_h = MixtureHamiltonian(
            list(self.hamiltonians),
            [1.0 for h in self.hamiltonians],
        )
        coefficients = all_h.get_coefficients(1.0 * hamiltonian)
        names = potential_component_names(len(self.hamiltonians))
        values = [self._data[name] for name in names]
        return add_contributions(
            coefficients,
            *values,
        )

    def get_state(self):
        return get_state(self.state, self.status)

    def save(self, path: Union[str, Path]):
        if type(path) is str:
            path = Path(path)
        assert not path.exists()
        raise NotImplementedError
