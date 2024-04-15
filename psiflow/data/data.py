from __future__ import annotations  # necessary for type-guarding class methods

from pathlib import Path
from typing import Optional, Union

import numpy as np
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data.geometry import (
    Geometry,
    assign_identifiers,
    extract_quantities,
    read_frames,
    write_frames,
)
from psiflow.data.utils import (
    align_axes,
    apply_offset,
    clean_frames,
    count_frames,
    get_elements,
    get_train_valid_indices,
    join_frames,
    not_null,
    reset_frames,
)
from psiflow.utils import copy_data_future, resolve_and_check, unpack_i


@psiflow.serializable
class Dataset:
    extxyz: psiflow._DataFuture

    def __init__(
        self,
        states: Optional[list[Union[AppFuture, Geometry]]],
        extxyz: Optional[psiflow._DataFuture] = None,
    ):
        if extxyz is not None:
            assert states is None
            self.extxyz = extxyz
        else:
            assert states is not None
            self.extxyz = write_frames(
                *states,
                outputs=[psiflow.context().new_file("data_", ".xyz")],
            ).outputs[0]

    def length(self) -> AppFuture:
        return count_frames(inputs=[self.extxyz])

    def shuffle(self):  # blocking
        indices = np.arange(self.length().result())
        np.random.shuffle(indices)
        return self.__getitem__([int(i) for i in indices])

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

    def get(self, *quantities: str):
        result = extract_quantities(
            quantities,
            None,
            None,
            inputs=[self.extxyz],
        )
        return tuple([unpack_i(result, i) for i in range(len(quantities))])

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

    def split(self, fraction):  # auto-shuffles
        train, valid = get_train_valid_indices(
            self.length(),
            fraction,
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
