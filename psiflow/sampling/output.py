import re
from pathlib import Path
from typing import Union

import numpy as np
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms
from psiflow.utils import unpack_i


def read_output(filename):
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


def _parse_data(
    keys: list[str],
    inputs: list = [],
) -> dict[str, np.ndarray]:
    from psiflow.sampling.output import read_output

    values, _ = read_output(inputs[0].filepath)
    bare_keys = []
    for key in keys:
        if "{" in key:
            bare_key = key.split("{")[0]
        else:
            bare_key = key
        bare_keys.append(bare_key)
    return [values[key] for key in bare_keys]


parse_data = python_app(_parse_data, executors=["default_threads"])


class SimulationOutput:
    def __init__(self, fields: list[str]):
        self._data = {key: None for key in fields}

        self.state = None
        self.stdout = None
        self.reset = None
        self.time = None
        self.trajectory = None

    def __getitem__(self, key: str):
        if key not in self._data:
            raise ValueError("output {} not available".format(key))
        return self._data[key]

    def load_output(self, state: Union[AppFuture, FlowAtoms]):
        pass

    def parse_data(self, data_future: DataFuture):
        data = parse_data(
            list(self._data.keys()),
            inputs=[data_future],
        )
        for i, key in enumerate(self._data.keys()):
            self._data[key] = unpack_i(data, i)

    def save(self, path: Union[str, Path]):
        if type(path) is str:
            path = Path(path)
        assert not path.exists()
        raise NotImplementedError
