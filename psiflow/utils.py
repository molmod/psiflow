from __future__ import annotations  # necessary for type-guarding class methods

import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.executors import WorkQueueExecutor
from parsl.launchers.base import Launcher


@typeguard.typechecked
def get_attribute(obj: Any, *attribute_names: str) -> Any:
    for name in attribute_names:
        obj = getattr(obj, name)
    return obj


@typeguard.typechecked
def _boolean_or(*args: Union[bool, np.bool_]) -> bool:
    return any(args)


boolean_or = python_app(_boolean_or, executors=["default_threads"])


def _multiply(a, b):
    return a * b


multiply = python_app(_multiply, executors=["default_threads"])


@typeguard.typechecked
def setup_logger(module_name):
    # Create logger instance for the module
    module_logger = logging.getLogger(module_name)

    # Set the desired format string
    formatter = logging.Formatter("%(name)s - %(message)s")

    # Create handler to send logs to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    # Add handler to the logger instance
    module_logger.addHandler(stdout_handler)

    # Set the logging level for the logger
    module_logger.setLevel(logging.INFO)

    return module_logger


def _compute_sum(a, b):
    return np.add(a, b)


compute_sum = python_app(_compute_sum, executors=["default_threads"])


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
        return None
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


@typeguard.typechecked
def _log_message(logger, message, *futures):
    if len(futures) > 0:
        logger.info(message.format(*futures))
    else:
        logger.info(message)


log_message = python_app(_log_message, executors=["default_threads"])


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

    input_dict = dict(input_dict)
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
def _load_metrics(inputs: list = []) -> np.recarray:
    return np.load(inputs[0], allow_pickle=True)


load_metrics = python_app(_load_metrics, executors=["default_threads"])


@typeguard.typechecked
def _save_metrics(data: np.recarray, outputs: list = []) -> None:
    with open(outputs[0], "wb") as f:
        data.dump(f)


save_metrics = python_app(_save_metrics, executors=["default_threads"])


@typeguard.typechecked
def _concatenate(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate(arrays)


concatenate = python_app(_concatenate, executors=["default_threads"])


@typeguard.typechecked
def container_launch_command(
    uri: str,
    engine: str = "apptainer",
    gpu: bool = False,
    addopts: str = " --no-eval -e --no-mount home -W /tmp --writable-tmpfs",
    entrypoint: str = "/opt/entry.sh",
) -> str:
    assert engine in ["apptainer", "singularity"]
    assert len(uri) > 0

    launch_command = ""
    launch_command += engine
    launch_command += " exec "
    launch_command += addopts
    launch_command += " --bind {}".format(
        Path.cwd().resolve()
    )  # access to data / internal dir
    if gpu:
        if "rocm" in uri:
            launch_command += " --rocm"
        else:  # default
            launch_command += " --nv"
    launch_command += " {} {} ".format(uri, entrypoint)
    return launch_command


class SlurmLauncher(Launcher):
    def __init__(self, debug: bool = True, overrides: str = ""):
        super().__init__(debug=debug)
        self.overrides = overrides

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        x = """set -e

NODELIST=$(scontrol show hostnames)
NODE_ARRAY=($NODELIST)
NODE_COUNT=${{#NODE_ARRAY[@]}}
EXPECTED_NODE_COUNT={nodes_per_block}

# Check if the length of NODELIST matches the expected number of nodes
if [ $NODE_COUNT -ne $EXPECTED_NODE_COUNT ]; then
  echo "Error: Expected $EXPECTED_NODE_COUNT nodes, but got $NODE_COUNT nodes."
  exit 1
fi

for NODE in $NODELIST; do
  srun --nodes=1 --ntasks=1 --exact -l {overrides} --nodelist=$NODE {command} &
  if [ $? -ne 0 ]; then
    echo "Command failed on node $NODE"
  fi
done

wait
""".format(
            nodes_per_block=nodes_per_block,
            command=command,
            overrides=self.overrides,
        )
        return x


class MyWorkQueueExecutor(WorkQueueExecutor):
    def _get_launch_command(self, block_id):
        return self.worker_command
