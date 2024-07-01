from __future__ import annotations  # necessary for type-guarding class methods

import math
import logging
from typing import Union, Optional, Type, Callable

import typeguard
import numpy as np
from parsl.dataflow.futures import AppFuture
from parsl.app.app import python_app, join_app
from parsl.app.python import PythonApp

import psiflow
from psiflow.data import Dataset, _read_frames, batch_frames
from psiflow.geometry import Geometry, create_outputs, NullState
from psiflow.functions import Function

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def _concatenate_multiple(*args: list[np.ndarray]) -> list[np.ndarray]:
    narrays = len(args[0])
    for arg in args:
        assert isinstance(arg, list)
    assert all([len(a) == narrays for a in args])

    concatenated = []
    for i in range(narrays):
        concatenated.append(np.concatenate([arg[i] for arg in args]))
    return concatenated


concatenate_multiple = python_app(_concatenate_multiple, executors=['default_threads'])


@typeguard.typechecked
def _aggregate_multiple(
    coefficients: np.ndarray,
    inputs: list = [],
) -> list[np.ndarray]:
    ncomponents = len(coefficients)
    assert len(inputs) % ncomponents == 0
    narrays = len(inputs) // ncomponents

    results = [np.zeros(inputs[_].shape) for _ in range(narrays)]
    for i in range(ncomponents):
        for j in range(narrays):
            results[j] += coefficients[i] * inputs[i * ncomponents + j]
    return results


aggregate_multiple = python_app(_aggregate_multiple, executors=['default_threads'])


@join_app
@typeguard.typechecked
def batch_apply(
    apply_app: Union[PythonApp, Callable],
    arg: Union[Dataset, list[Geometry]],
    batch_size: int,
    length: int,
    outputs: list = [],
    reduce_func: Optional[PythonApp] = None,
    **app_kwargs,
) -> AppFuture:
    nbatches = math.ceil(length / batch_size)
    batches = [psiflow.context().new_file("data_", ".xyz") for _ in range(nbatches)]
    future = batch_frames(batch_size, inputs=[arg.extxyz], outputs=batches)
    output_futures = []
    for i in range(nbatches):
        f = apply_app(
            None,
            inputs=[future.outputs[i]],
            **app_kwargs,
        )
        output_futures.append(f)
    future = concatenate_multiple(*output_futures)
    return future


@python_app(executors=['default_threads'])
def get_length(arg):
    if isinstance(arg, list):
        return len(arg)
    else:
        return 1


@staticmethod
def sort_outputs(
    outputs_: list[str],
    **kwargs,
) -> list[np.ndarray]:
    output_arrays = []
    for name in outputs_:
        array = kwargs.get(name, None)
        assert array is not None
        output_arrays.append(array)
    return output_arrays


def _apply(
    arg: Union[Geometry, list[Geometry], None],
    outputs_: tuple[str, ...],
    inputs: list = [],
    function_cls: Optional[Type[Function]] = None,
    **parameters,
) -> Optional[list[np.ndarray]]:
    assert function_cls is not None
    if arg is None:
        states = _read_frames(inputs=[inputs[0]])
    elif not isinstance(arg, list):
        states = [arg]
    else:
        states = arg
    function = function_cls(**parameters)
    output_dict = function(states)
    output_arrays = sort_outputs(outputs_, **output_dict)
    return output_arrays


def compute(
    apply_app: PythonApp,
    arg: Union[Dataset, AppFuture[list], list[Union[AppFuture, Geometry]]],
    outputs_: Optional[list[str]] = None,
    batch_size: Optional[int] = None,
) -> Union[list[AppFuture], AppFuture]:
    if batch_size is not None:
        if isinstance(arg, Dataset):
            length = arg.length()
        else:
            length = get_length(arg)
            # convert to Dataset for convenience
            arg = Dataset(arg)
        future = batch_apply(
            apply_app,
            arg,
            batch_size,
            length,
            outputs_=outputs_,
        )
    else:
        if isinstance(arg, Dataset):
            future = apply_app(
                None,
                outputs_=outputs_,
                inputs=[arg.extxyz],
            )
        else:
            future = apply_app(
                arg,
                outputs_=outputs_,
                inputs=[],
            )
    if len(outputs_) == 1:
        return future[0]
    else:
        return [future[i] for i in range(len(outputs_))]
