from __future__ import annotations  # necessary for type-guarding class methods

import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Union

import numpy as np
import typeguard
from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def _multiply(x: Any, by: float) -> Any:
    return by * x


multiply = python_app(_multiply, executors=["default_threads"])


@typeguard.typechecked
def set_logger(  # hacky
    level: Union[str, int],  # 'DEBUG' or logging.DEBUG
):
    formatter = logging.Formatter(fmt="%(levelname)s - %(name)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    names = [
        "psiflow.data",
        "psiflow.committee",
        "psiflow.metrics",
        "psiflow.execution",
        "psiflow.state",
        "psiflow.learning",
        "psiflow.learning_utils",
        "psiflow.utils",
        "psiflow.parsl_utils",
        "psiflow.models.model",
        "psiflow.models.mace",
        "psiflow.reference._cp2k",
        "psiflow.reference._emt",
        "psiflow.reference._pyscf",
    ]
    for name in names:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)


@typeguard.typechecked
def _sum_integers(a: int, b: int) -> int:
    return a + b


sum_integers = python_app(_sum_integers, executors=["default_threads"])


@typeguard.typechecked
def _combine_futures(inputs: list[Any]) -> list[Any]:
    return list(inputs)


combine_futures = python_app(_combine_futures, executors=["default_threads"])


@typeguard.typechecked
def _dump_json(
    inputs: list = [],
    outputs: list = [],
    **kwargs,
) -> None:
    import json

    import numpy as np

    def convert_to_list(array):
        if not type(array) is np.ndarray:
            return array
        as_list = []
        for item in array:
            as_list.append(convert_to_list(item))
        return as_list

    for name in list(kwargs.keys()):
        value = kwargs[name]
        if type(value) is np.ndarray:
            value = convert_to_list(value)
        kwargs[name] = value
    with open(outputs[0], "w") as f:
        f.write(json.dumps(kwargs))


dump_json = python_app(_dump_json, executors=["default_threads"])


@typeguard.typechecked
def _copy_data_future(
    pass_on_exist: bool = False,
    inputs: list[File] = [],
    outputs: list[File] = [],
) -> None:
    import shutil
    from pathlib import Path

    assert len(inputs) == 1
    assert len(outputs) == 1
    if Path(outputs[0]).is_file() and pass_on_exist:
        return 0
    if Path(inputs[0]).is_file():
        shutil.copyfile(inputs[0], outputs[0])
    else:  # no need to copy empty file
        pass


copy_data_future = python_app(_copy_data_future, executors=["default_threads"])


@typeguard.typechecked
def _copy_app_future(future: Any, inputs: list = [], outputs: list = []) -> Any:
    # inputs/outputs to enforce additional dependencies
    from copy import deepcopy

    return deepcopy(future)


copy_app_future = python_app(_copy_app_future, executors=["default_threads"])


@join_app
@typeguard.typechecked
def log_message(message, *futures):
    logger.info(message.format(*futures))
    return copy_app_future(0)


def _pack(*args):
    return args


pack = python_app(_pack, executors=["default_threads"])


@typeguard.typechecked
def _unpack_i(result: Union[np.ndarray, list, tuple], i: int) -> Any:
    assert i <= len(result)
    return result[i]


unpack_i = python_app(_unpack_i, executors=["default_threads"])


@typeguard.typechecked
def _save_yaml(
    input_dict: dict,
    outputs: list[File] = [],
    **extra_keys: Any,
) -> None:
    import yaml

    def _make_dict_safe(arg):
        # walks through dict and converts numpy types to python natives
        for key in list(arg.keys()):
            if hasattr(arg[key], "item"):
                arg[key] = arg[key].item()
            elif type(arg[key]) is dict:
                arg[key] = _make_dict_safe(arg[key])
            else:
                pass
        return arg

    for key, value in extra_keys.items():
        assert key not in input_dict
        input_dict[key] = value
    input_dict = _make_dict_safe(input_dict)
    with open(outputs[0], "w") as f:
        yaml.dump(input_dict, f, default_flow_style=False)


save_yaml = python_app(_save_yaml, executors=["default_threads"])


@typeguard.typechecked
def _save_xml(
    element: ET.Element,
    outputs: list = [],
) -> None:
    tree = ET.ElementTree(element)
    ET.indent(tree, "  ")
    tree.write(outputs[0], encoding="utf-8", xml_declaration=True)


save_xml = python_app(_save_xml, executors=["default_threads"])


@typeguard.typechecked
def _load_numpy(inputs: list[File] = [], **kwargs) -> np.ndarray:
    return np.loadtxt(inputs[0], **kwargs)


load_numpy = python_app(_load_numpy, executors=["default_threads"])


@typeguard.typechecked
def _read_yaml(inputs: list[File] = [], outputs: list[File] = []) -> dict:
    import yaml

    with open(inputs[0], "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict


read_yaml = python_app(_read_yaml, executors=["default_threads"])


@typeguard.typechecked
def _save_txt(data: str, outputs: list[File] = []) -> None:
    with open(outputs[0], "w") as f:
        f.write(data)


save_txt = python_app(_save_txt, executors=["default_threads"])


@typeguard.typechecked
def resolve_and_check(path: Path) -> Path:
    path = path.resolve()
    if Path.cwd() in path.parents:
        pass
    elif path.exists() and Path.cwd().samefile(path):
        pass
    else:
        raise ValueError(
            "requested file and/or path at location: {}"
            "\nwhich is not in the present working directory: {}"
            "\npsiflow can only load and/or save in its present "
            "working directory because this is the only directory"
            " that will get bound into the container.".format(path, Path.cwd())
        )
    return path


@typeguard.typechecked
def apply_temperature_ramp(
    T_min: float, T_max: float, nsteps: int, current_temperature: float
) -> float:
    assert T_max > T_min
    if nsteps > 1:
        delta_beta = (1 / T_min - 1 / T_max) / (nsteps - 1)
        next_beta = 1 / current_temperature - delta_beta
        if (next_beta > 0) and (next_beta > 1 / T_max):
            return 1 / next_beta
        else:
            return T_max
    else:
        return T_max


@typeguard.typechecked
def _load_metrics(inputs: list = []) -> np.recarray:
    return np.load(inputs[0], allow_pickle=True)


load_metrics = python_app(_load_metrics, executors=["default_threads"])


@typeguard.typechecked
def _save_metrics(data: np.recarray, outputs: list = []) -> None:
    with open(outputs[0], "wb") as f:
        data.dump(f)


save_metrics = python_app(_save_metrics, executors=["default_threads"])
