import shutil
import textwrap
from copy import deepcopy
from typing import Any, Union, Callable
from pathlib import Path

import numpy as np
from parsl import python_app
from parsl.data_provider.files import File


def get_attribute(obj: Any, *attribute_names: str) -> Any:
    # uses Parsl lifted operators in case obj is a Future
    for name in attribute_names:
        obj = getattr(obj, name)
    return obj


@python_app(executors=["default_threads"])
def multiply(a, b) -> float:
    return a * b


@python_app(executors=["default_threads"])
def compute_sum(a, b) -> float:
    return np.add(a, b)


@python_app(executors=["default_threads"])
def copy_data_future(
    file: str | Path | File,
    pass_on_exist: bool = False,
    inputs: list = [],
    outputs: list[File] = []
) -> None:
    """Copy file to new location once all input futures complete"""
    assert len(outputs) == 1
    file = Path(file)
    file_out = Path(outputs[0])
    if file == file_out:
        return  # no copy needed
    if file_out.is_file() and pass_on_exist:
        return
    if not file.is_file():
        return  # no need to copy empty file

    shutil.copyfile(file, file_out)


@python_app(executors=["default_threads"])
def copy_app_future(future: Any, inputs: list = []) -> Any:
    """Return a deepcopy once all input futures complete"""
    return deepcopy(future)


@python_app(executors=["default_threads"])
def log_message(logging_func: Callable, message: str, *futures, inputs: list = []) -> None:
    """Delay a logging call until all futures complete"""
    if len(futures) > 0:
        message = message.format(*futures)
    logging_func(message)


@python_app(executors=["default_threads"])
def pack(*args: Any) -> tuple[Any]:
    """Combine passed futures into a single future."""
    return args


def create_bash_template(tmpdir_root: str, keep_tmpdirs: bool) -> str:
    """Create general wrapper for all bash apps. The exitcode ensures that every app completes successfully."""
    # TODO: does not belong here
    template = f"""
    # Create and move into new tmpdir for app execution
    tmpdir=$(mktemp -d -p {tmpdir_root} "psiflow-tmp.XXXXXXXXXX")
    cd $tmpdir; echo "tmpdir: $PWD"
    {{env}}
    # printenv

    # Actual app definition goes here
    {{commands}}

    # Cleanup
    {'cd ../.. && rm -r $tmpdir' if not keep_tmpdirs else ''}
    exit 0
    """
    return textwrap.dedent(template)
