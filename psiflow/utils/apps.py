from __future__ import annotations  # necessary for type-guarding class methods

import logging
import sys
from typing import Any, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.data_provider.files import File


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
def _concatenate(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate(arrays)


concatenate = python_app(_concatenate, executors=["default_threads"])


@typeguard.typechecked
def _isnan(a: Union[float, np.ndarray]) -> bool:
    return bool(np.any(np.isnan(a)))


isnan = python_app(_isnan, executors=["default_threads"])
