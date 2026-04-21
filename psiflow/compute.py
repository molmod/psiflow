from pathlib import Path
from typing import Callable, ClassVar, Optional, Union, Type, Any, TypeAlias
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from parsl.app.app import join_app, python_app
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.geometry import Geometry, PER_ATOM_FIELDS, DEFAULT_PROPERTIES, MISSING
from psiflow.data import Dataset
from psiflow.data.utils import insert
from psiflow.functions import Function
from psiflow.utils.apps import pack

ComputeInput: TypeAlias = (
    Dataset | list[Geometry] | list[AppFuture] | AppFuture | Geometry
)


@dataclass
class ComputeResult:
    """Container to hold and manipulate the results from compute/apply operations."""

    n_atoms: np.ndarray
    data: dict[str, np.ndarray]

    def __post_init__(self):
        self.cutoffs = self.n_atoms.cumsum()

    def __getattr__(self, item):
        """Enable parsl deferred_getitem on AppFuture"""
        if item == "data":
            raise AttributeError()  # prevent RecursionError from pickle
        return self.data[item]

    @property
    def keys(self) -> set[str]:
        return set(self.data.keys())

    def get(self, key: str, per_geom: bool = False) -> list | np.ndarray:
        arr = self.data[key]
        if not per_geom:
            return arr

        if self.cutoffs[-1] == arr.shape[0]:
            data_list = np.array_split(arr, self.cutoffs[:-1])
        else:
            data_list = list(arr)
        return data_list

    def to_dict(self) -> dict[str, list]:
        """Convert to a dict that is extract/insert compliant for geometries"""
        return {k: self.get(k, per_geom=True) for k in self.keys}

    @classmethod
    def from_data(cls, n_atoms: np.ndarray, data: dict[str, list]):
        values = {}
        for k, v in data.items():
            assert len(v) == n_atoms.size
            if not np.iterable(v[0]):
                values[k] = np.array(v)
            elif len(v[0]) == n_atoms[0] and len(v[-1]) == n_atoms[-1]:
                values[k] = np.concatenate(v)  # assume per-atom property
            else:
                values[k] = np.stack(v)
        return cls(np.array(n_atoms), values)


def _apply(
    states: Sequence[Geometry],
    function_cls: Type[Function],
    inputs: Sequence = (),
    parsl_resource_specification: dict = {},
    **parameters,
) -> ComputeResult:
    assert function_cls is not None
    function = function_cls(**parameters)  # psiflow.functions.Function subclass
    output_dict = function.compute(states)
    n_atoms = np.array([len(geom) for geom in states])
    return ComputeResult.from_data(n_atoms, output_dict)


@python_app(executors=["default_threads"])
def concatenate_results(*results: ComputeResult) -> ComputeResult:
    """"""
    n_atoms_list = [result.n_atoms for result in results]
    n_atoms = np.concatenate(n_atoms_list)

    data = {}
    for k in results[0].keys:
        values = [result.get(k) for result in results]
        data[k] = np.concatenate(values)

    return ComputeResult(n_atoms, data)


@python_app(executors=["default_threads"])
def aggregate_results(
    *results: ComputeResult, coefficients: Optional[np.ndarray] = None
) -> ComputeResult:
    """"""
    if coefficients is None:
        coefficients = np.ones(len(results))
    assert len(coefficients) == len(results)

    n_atoms = results[0].n_atoms
    for result in results[1:]:
        assert np.allclose(n_atoms, result.n_atoms)

    data = {k: np.zeros_like(v) for k, v in results[0].data.items()}
    for i, result in enumerate(results):
        for k in data:
            data[k] += result.get(k) * coefficients[i]

    return ComputeResult(n_atoms, data)


@join_app
def batch_apply(
    states: list[Geometry],
    apply_apps: Sequence[Callable],
    batch_size: int,
    reduce_func: Callable,
) -> AppFuture:
    """Apply a set of apps to batches of data"""
    # TODO: holds everything in memory -- what with very large datasets?
    n_batch = len(states) // batch_size + 1 * (len(states) % batch_size > 0)
    batches = [arr.tolist() for arr in np.array_split(states, n_batch)]

    output = []
    for batch in batches:
        futures = []
        for app in apply_apps:
            future = app(batch)
            futures.append(future)
        future = reduce_func(*futures)
        output.append(future)

    return concatenate_results(*output)


def compute(
    arg: ComputeInput,
    apply_apps: Sequence[Callable],
    reduce_func: Union[python_app, Callable] = aggregate_results,
    batch_size: Optional[int] = None,
) -> AppFuture:
    """
    Compute results by applying apps to the input data.

    Args:
        arg: Input data to compute on.
        apply_apps: Apps to apply to the data.
        reduce_func: Function to reduce results.
        batch_size: Optional batch size for processing.
    """
    states = input_to_geometries(arg)
    if batch_size is None:
        futures = []
        for app in apply_apps:
            future = app(states)
            futures.append(future)
        future = reduce_func(*futures)
    else:
        future = batch_apply(states, apply_apps, batch_size, reduce_func)

    return future


metric_func_map: dict[str, Callable] = {
    "RMSE": lambda arr1, arr2: np.mean((arr1 - arr2) ** 2) ** 0.5,
    "MAE": lambda arr1, arr2: np.abs(arr1 - arr2).mean(),
}


def _compare_results(
    result1: ComputeResult,
    result2: Optional[ComputeResult] = None,
    metric: str = "RMSE",
    reduce: bool = True,
    **kwargs: list | np.ndarray,
) -> dict:
    """"""
    # TODO: what with missing values?
    assert (result2 is None) != (len(kwargs) == 0)  # xor
    if kwargs:
        result2 = ComputeResult.from_data(result1.n_atoms, kwargs)
    elif not np.allclose(result1.cutoffs, result2.cutoffs):
        raise ValueError("Results cannot be compared")

    metric_func = metric_func_map[metric]
    keys = result1.keys.intersection(result2.keys)

    out = {}
    for k in keys:
        if reduce:
            arr1, arr2 = result1.get(k), result2.get(k)
            out[k] = metric_func(arr1, arr2)
        else:
            list1, list2 = result1.get(k, per_geom=True), result2.get(k, per_geom=True)
            out[k] = [metric_func(v1, v2) for v1, v2 in zip(list1, list2)]

    return out


compare_results = python_app(_compare_results, executors=["default_threads"])


def input_to_geometries(data: ComputeInput) -> AppFuture:
    """Convert ComputeInput into a sequence of geometries (as a future)"""
    # Dataset | list[Geometry] | list[AppFuture] | AppFuture | Geometry

    @python_app(executors=["default_threads"])
    def prep_input(data: Geometry | Sequence[Geometry]) -> list[Geometry]:
        # make sure apply apps get consistent input
        if isinstance(data, Geometry):
            data = [data]
        return data

    if isinstance(data, Dataset):
        return data.geometries()
    elif isinstance(data, Geometry):
        return pack(data)
    elif isinstance(data, AppFuture):
        return prep_input(data)
    elif isinstance(data, list):
        return pack(*data)
    else:
        assert False


@python_app(executors=["default_threads"])
def insert_results(
    states: Sequence[Geometry], result: ComputeResult
) -> Sequence[Geometry]:
    """"""
    insert(states, result.to_dict())
    return states
