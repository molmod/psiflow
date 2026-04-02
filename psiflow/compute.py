import math
from typing import Callable, ClassVar, Optional, Union

import numpy as np
import typeguard
from parsl import python_app
from parsl.app.app import join_app, python_app
from parsl.app.python import PythonApp
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.data import Dataset
from psiflow.data.utils import batch_frames


def _concatenate_multiple(*args: list[np.ndarray]) -> list[np.ndarray]:
    """
    Concatenate multiple lists of arrays.

    Args:
        *args: Lists of numpy arrays to concatenate.

    Returns:
        list[np.ndarray]: List of concatenated arrays.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """

    def pad_arrays(
        arrays: list[np.ndarray],
        pad_dimension: int = 1,
    ) -> list[np.ndarray]:
        ndims = np.array([len(a.shape) for a in arrays])
        assert np.all(ndims == ndims[0])
        assert np.all(pad_dimension < ndims)

        pad_size = max([a.shape[pad_dimension] for a in arrays])
        for i in range(len(arrays)):
            shape = list(arrays[i].shape)
            shape[pad_dimension] = pad_size - shape[pad_dimension]
            padding = np.zeros(tuple(shape)) + np.nan
            arrays[i] = np.concatenate((arrays[i], padding), axis=pad_dimension)
        return arrays

    narrays = len(args[0])
    for arg in args:
        assert isinstance(arg, list)
    assert all([len(a) == narrays for a in args])

    concatenated = []
    for i in range(narrays):
        arrays = [arg[i] for arg in args]
        if len(arrays[0].shape) > 1:
            pad_arrays(arrays)
        concatenated.append(np.concatenate(tuple(arrays)))
    return concatenated


concatenate_multiple = python_app(_concatenate_multiple, executors=["default_threads"])


def _aggregate_multiple(
    *arrays_list,
    coefficients: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """
    Aggregate multiple lists of arrays with optional coefficients.

    Args:
        *arrays_list: Lists of arrays to aggregate.
        coefficients: Optional coefficients for weighted aggregation.

    Returns:
        list[np.ndarray]: List of aggregated arrays.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    if coefficients is None:
        coefficients = np.ones(len(arrays_list))
    else:
        assert len(coefficients) == len(arrays_list)

    results = [np.zeros(a.shape) for a in arrays_list[0]]
    for i, arrays in enumerate(arrays_list):
        for j, array in enumerate(arrays):
            results[j] += coefficients[i] * array
    return results


aggregate_multiple = python_app(_aggregate_multiple, executors=["default_threads"])


@join_app
def batch_apply(
    apply_apps: tuple[Union[PythonApp, Callable]],
    arg: Union[Dataset, list[Geometry]],
    batch_size: int,
    length: int,
    outputs: list = [],
    reduce_func: Optional[PythonApp] = None,
    **app_kwargs,
) -> AppFuture:
    """
    Apply a set of apps to batches of data.

    Args:
        apply_apps: Tuple of PythonApps or Callables to apply.
        arg: Dataset or list of Geometries to process.
        batch_size: Size of each batch.
        length: Total number of items to process.
        outputs: List of output files.
        reduce_func: Optional function to reduce results.
        **app_kwargs: Additional keyword arguments for the apps.

    Returns:
        AppFuture: Future representing the result of batch application.

    Note:
        This function is wrapped as a Parsl join_app.
    """
    nbatches = math.ceil(length / batch_size)
    batches = [psiflow.context().new_file("data_", ".xyz") for _ in range(nbatches)]
    future = batch_frames(batch_size, inputs=[arg.extxyz], outputs=batches)
    output_futures = []
    for i in range(nbatches):
        futures = []
        for app in apply_apps:
            f = app(
                None,
                inputs=[future.outputs[i]],
                **app_kwargs,
            )
            futures.append(f)
        reduced = reduce_func(*futures)
        output_futures.append(reduced)
    future = concatenate_multiple(*output_futures)
    return future


@python_app(executors=["default_threads"])
def get_length(arg):
    """
    Get the length of the input argument.

    Args:
        arg: Input to get the length of.

    Returns:
        int: Length of the input.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    if isinstance(arg, list):
        return len(arg)
    else:
        return 1


def compute(
    arg: Union[Dataset, AppFuture[list], list, AppFuture, Geometry],
    *apply_apps: Union[PythonApp, Callable],
    outputs_: Union[str, list[str], tuple[str, ...], None] = None,
    reduce_func: Union[PythonApp, Callable] = aggregate_multiple,
    batch_size: Optional[int] = None,
) -> Union[list[AppFuture], AppFuture]:
    """
    Compute results by applying apps to the input data.

    Args:
        arg: Input data to compute on.
        *apply_apps: Apps to apply to the data.
        outputs_: Names of output quantities.
        reduce_func: Function to reduce results.
        batch_size: Optional batch size for processing.

    Returns:
        Union[list[AppFuture], AppFuture]: Future(s) representing computation results.
    """
    if type(outputs_) is str:
        outputs_ = [outputs_]
    if batch_size is not None:
        if isinstance(arg, Dataset):
            length = arg.length()
        else:
            length = get_length(arg)
            # convert to Dataset for convenience
            arg = Dataset(arg)
        future = batch_apply(
            apply_apps,
            arg,
            batch_size,
            length,
            outputs_=outputs_,
            reduce_func=reduce_func,
        )
    else:
        futures = []
        if isinstance(arg, Dataset):
            for app in apply_apps:
                future = app(
                    None,
                    outputs_=outputs_,
                    inputs=[arg.extxyz],
                )
                futures.append(future)
        else:
            for app in apply_apps:
                future = app(
                    arg,
                    outputs_=outputs_,
                    inputs=[],
                )
                futures.append(future)
        future = reduce_func(*futures)
    if len(outputs_) == 1:
        return future[0]
    else:
        return [future[i] for i in range(len(outputs_))]


class Computable:
    """
    Base class for computable objects.

    Attributes:
        outputs (ClassVar[tuple[str, ...]]): Names of output quantities.
        batch_size (ClassVar[Optional[int]]): Default batch size for computation.
    """

    outputs: ClassVar[tuple[str, ...]] = ()
    batch_size: ClassVar[Optional[int]] = None

    def compute(
        self,
        arg: Union[Dataset, AppFuture[list], list, AppFuture, Geometry],
        *outputs: Optional[str],
        batch_size: Optional[int] = -1,  # if -1: take class default
    ) -> Union[list[AppFuture], AppFuture]:
        """
        Compute results for the given input.

        Args:
            arg: Input data to compute on.
            outputs: Names of output quantities.
            batch_size: Batch size for computation.

        Returns:
            Union[list[AppFuture], AppFuture]: Future(s) representing computation results.
        """
        raise NotImplementedError


@typeguard.typechecked
def _compute_rmse(
    array0: np.ndarray,
    array1: np.ndarray,
    reduce: bool = True,
) -> Union[float, np.ndarray]:
    """
    Compute the Root Mean Square Error (RMSE) between two arrays.

    Args:
        array0: First array.
        array1: Second array.
        reduce: Whether to reduce the result to a single value.

    Returns:
        Union[float, np.ndarray]: RMSE value(s).

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert array0.shape == array1.shape
    assert np.all(np.isnan(array0) == np.isnan(array1))

    se = (array0 - array1) ** 2
    se = se.reshape(se.shape[0], -1)

    if reduce:  # across both dimensions
        mask = np.logical_not(np.isnan(se))
        return float(np.sqrt(np.mean(se[mask])))
    else:  # retain first dimension
        if se.ndim == 1:
            return se
        else:
            values = np.empty(len(se))
            for i in range(len(se)):
                if np.all(np.isnan(se[i])):
                    values[i] = np.nan
                else:
                    mask = np.logical_not(np.isnan(se[i]))
                    value = np.sqrt(np.mean(se[i][mask]))
                    values[i] = value
            return values


compute_rmse = python_app(_compute_rmse, executors=["default_threads"])


@typeguard.typechecked
def _compute_mae(
    array0,
    array1,
    reduce: bool = True,
) -> Union[float, np.ndarray]:
    """
    Compute the Mean Absolute Error (MAE) between two arrays.

    Args:
        array0: First array.
        array1: Second array.
        reduce: Whether to reduce the result to a single value.

    Returns:
        Union[float, np.ndarray]: MAE value(s).

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert array0.shape == array1.shape
    mask0 = np.logical_not(np.isnan(array0))
    mask1 = np.logical_not(np.isnan(array1))
    assert np.all(mask0 == mask1)
    ae = np.abs(array0 - array1)
    to_reduce = tuple(range(1, array0.ndim))
    mask = np.logical_not(np.all(np.isnan(ae), axis=to_reduce))
    ae = ae[mask0].reshape(np.sum(1 * mask), -1)
    if reduce:  # across both dimensions
        return float(np.sqrt(np.mean(ae)))
    else:  # retain first dimension
        return np.sqrt(np.mean(ae, axis=1))


compute_mae = python_app(_compute_mae, executors=["default_threads"])
