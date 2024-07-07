from __future__ import annotations  # necessary for type-guarding class methods

import math
from pathlib import Path
from typing import Callable, ClassVar, Optional, Union

import numpy as np
import typeguard
from parsl.app.app import join_app, python_app
from parsl.app.python import PythonApp
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import QUANTITIES, Geometry
from psiflow.utils import combine_futures, copy_data_future, resolve_and_check, unpack_i

from .utils import (
    align_axes,
    app_filter,
    apply_offset,
    assign_identifiers,
    batch_frames,
    clean_frames,
    count_frames,
    extract_quantities,
    get_elements,
    get_train_valid_indices,
    insert_quantities,
    join_frames,
    not_null,
    read_frames,
    reset_frames,
    shuffle,
    write_frames,
)


@psiflow.serializable
class Dataset:
    extxyz: psiflow._DataFuture

    def __init__(
        self,
        states: Optional[list[Union[AppFuture, Geometry]], AppFuture],
        extxyz: Optional[psiflow._DataFuture] = None,
    ):
        if extxyz is not None:
            assert states is None
            self.extxyz = extxyz
        else:
            assert states is not None
            if not isinstance(states, list):  # AppFuture[list, geometry]
                extra_states = states
                states = []
            else:
                extra_states = None
            self.extxyz = write_frames(
                *states,
                extra_states=extra_states,
                outputs=[psiflow.context().new_file("data_", ".xyz")],
            ).outputs[0]

    def length(self) -> AppFuture:
        return count_frames(inputs=[self.extxyz])

    def shuffle(self):
        extxyz = shuffle(
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def __getitem__(
        self,
        index: Union[int, slice, list[int], AppFuture],
    ) -> Union[Dataset, AppFuture]:
        if isinstance(index, int):
            future = read_frames(
                [index],
                inputs=[self.extxyz],
                outputs=[],  # will return Geometry as Future
            )
            return unpack_i(future, 0)
        else:  # slice, list, AppFuture
            extxyz = read_frames(
                index,
                inputs=[self.extxyz],
                outputs=[psiflow.context().new_file("data_", ".xyz")],
            ).outputs[0]
            return Dataset(None, extxyz)

    def save(self, path: Union[Path, str]) -> AppFuture:
        path = resolve_and_check(Path(path))
        _ = copy_data_future(
            inputs=[self.extxyz],
            outputs=[File(str(path))],
        )

    def geometries(self) -> AppFuture:
        return read_frames(inputs=[self.extxyz])

    def __add__(self, dataset: Dataset) -> Dataset:
        extxyz = join_frames(
            inputs=[self.extxyz, dataset.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def subtract_offset(self, **atomic_energies: Union[float, AppFuture]) -> Dataset:
        assert len(atomic_energies) > 0
        extxyz = apply_offset(
            True,
            **atomic_energies,
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def add_offset(self, **atomic_energies) -> Dataset:
        assert len(atomic_energies) > 0
        extxyz = apply_offset(
            False,
            **atomic_energies,
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def elements(self):
        return get_elements(inputs=[self.extxyz])

    def reset(self):
        extxyz = reset_frames(
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def clean(self):
        extxyz = clean_frames(
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def get(
        self,
        *quantities: str,
        atom_indices: Optional[list[int]] = None,
        elements: Optional[list[str]] = None,
    ):
        result = extract_quantities(
            quantities,
            atom_indices,
            elements,
            inputs=[self.extxyz],
        )
        if len(quantities) == 1:
            return unpack_i(result, 0)
        else:
            return tuple([unpack_i(result, i) for i in range(len(quantities))])

    def set(
        self,
        quantity: str,
        array: Union[np.ndarray, AppFuture],
        atom_indices: Optional[list[int]] = None,
        elements: Optional[list[str]] = None,
    ):
        pass

    def evaluate(
        self,
        computable: Computable,
        batch_size: Optional[int] = None,
    ) -> Dataset:
        if batch_size is not None:
            outputs = computable.compute(self, batch_size=batch_size)
        else:
            outputs = computable.compute(self)  # use default from computable
        future = insert_quantities(
            quantities=computable.outputs,
            arrays=combine_futures(inputs=outputs),
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        )
        return Dataset(None, future.outputs[0])

    def filter(
        self,
        quantity: str,
    ) -> Dataset:
        assert quantity in QUANTITIES
        extxyz = app_filter(
            quantity,
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def not_null(self) -> Dataset:
        extxyz = not_null(
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def align_axes(self):
        extxyz = align_axes(
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def split(self, fraction, shuffle=True):  # auto-shuffles
        train, valid = get_train_valid_indices(
            self.length(),
            fraction,
            shuffle,
        )
        return self.__getitem__(train), self.__getitem__(valid)

    def assign_identifiers(
        self, identifier: Union[int, AppFuture, None] = None
    ) -> AppFuture:
        result = assign_identifiers(
            identifier,
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        )
        self.extxyz = result.outputs[0]
        return result

    @classmethod
    def load(
        cls,
        path_xyz: Union[Path, str],
    ) -> Dataset:
        path_xyz = resolve_and_check(Path(path_xyz))
        assert path_xyz.exists()  # needs to be locally accessible
        return cls(None, extxyz=File(str(path_xyz)))


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


concatenate_multiple = python_app(_concatenate_multiple, executors=["default_threads"])


@typeguard.typechecked
def _aggregate_multiple(
    *arrays_list,
    coefficients: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
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
@typeguard.typechecked
def batch_apply(
    apply_apps: tuple[Union[PythonApp, Callable]],
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
    if isinstance(arg, list):
        return len(arg)
    else:
        return 1


@typeguard.typechecked
def compute(
    arg: Union[Dataset, AppFuture[list], list, AppFuture, Geometry],
    *apply_apps: Union[PythonApp, Callable],
    outputs_: Union[str, list[str], None] = None,
    reduce_func: Union[PythonApp, Callable] = aggregate_multiple,
    batch_size: Optional[int] = None,
) -> Union[list[AppFuture], AppFuture]:
    if outputs_ is not None and not isinstance(outputs_, list):
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


@typeguard.typechecked
class Computable:
    outputs: ClassVar[tuple[str, ...]] = ()
    batch_size: ClassVar[Optional[int]] = None

    def compute(
        self,
        arg: Union[Dataset, AppFuture[list], list, AppFuture, Geometry],
        outputs: Union[str, list[str], None] = None,
        batch_size: Optional[int] = -1,  # if -1: take class default
    ) -> Union[list[AppFuture], AppFuture]:
        if outputs is None:
            outputs = list(self.__class__.outputs)
        if batch_size == -1:
            batch_size = self.__class__.batch_size
        return compute(
            arg,
            self.app,
            outputs_=outputs,
            batch_size=batch_size,
        )
