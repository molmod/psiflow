import shutil
import textwrap
from typing import Any, Union
from pathlib import Path

import numpy as np
from parsl import python_app
from parsl.data_provider.files import File


def get_attribute(obj: Any, *attribute_names: str) -> Any:
    # TODO: not an app
    for name in attribute_names:
        obj = getattr(obj, name)
    return obj


def _boolean_or(*args: Union[bool, np.bool_]) -> bool:
    return any(args)


boolean_or = python_app(_boolean_or, executors=["default_threads"])


def _multiply(a, b):
    return a * b


multiply = python_app(_multiply, executors=["default_threads"])


def _compute_sum(a, b):
    return np.add(a, b)


compute_sum = python_app(_compute_sum, executors=["default_threads"])


def _copy_data_future(
    pass_on_exist: bool = False,
    inputs: list[File] = [],
    outputs: list[File] = [],
) -> None:
    assert len(inputs) == 1
    assert len(outputs) == 1
    if Path(outputs[0]).is_file() and pass_on_exist:
        pass
    elif Path(inputs[0]).is_file():
        shutil.copyfile(inputs[0], outputs[0])
    else:  # no need to copy empty file
        pass
    return


copy_data_future = python_app(_copy_data_future, executors=["default_threads"])


def _copy_app_future(future: Any, inputs: list = [], outputs: list = []) -> Any:
    # inputs/outputs to enforce additional dependencies
    from copy import deepcopy

    return deepcopy(future)


copy_app_future = python_app(_copy_app_future, executors=["default_threads"])


def _log_message(logger, message, *futures):
    if len(futures) > 0:
        logger.info(message.format(*futures))
    else:
        logger.info(message)


log_message = python_app(_log_message, executors=["default_threads"])


@python_app(executors=["default_threads"])
def pack(*args: Any) -> tuple[Any]:
    """Combine passed futures into a single future."""
    return args


def _concatenate(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate(arrays)


concatenate = python_app(_concatenate, executors=["default_threads"])


def _isnan(a: Union[float, np.ndarray]) -> bool:
    return bool(np.any(np.isnan(a)))


isnan = python_app(_isnan, executors=["default_threads"])


def create_bash_template(tmpdir_root: str, keep_tmpdirs: bool) -> str:
    """Create general wrapper for all bash apps. The exitcode ensures that every app completes successfully."""
    template = f"""
    # Create and move into new tmpdir for app execution
    tmpdir=$(mktemp -d -p {tmpdir_root} "psiflow-tmp.XXXXXXXXXX")
    cd $tmpdir; echo "tmpdir: $PWD"
    {{env}}
    printenv

    # Actual app definition goes here
    {{commands}}

    # Cleanup
    {'cd ../.. && rm -r $tmpdir' if not keep_tmpdirs else ''}
    exit 0
    """
    return textwrap.dedent(template)

