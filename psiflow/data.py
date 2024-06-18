from __future__ import annotations  # necessary for type-guarding class methods

import math
import re
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from ase.data import atomic_numbers, chemical_symbols
from parsl.app.app import join_app, python_app
from parsl.app.python import PythonApp
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry, NullState, create_outputs, QUANTITIES
from psiflow.utils import copy_data_future, resolve_and_check, unpack_i


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
            if isinstance(states, AppFuture):  # Future of list
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

    def evaluate(
        self,
        function,
        outputs: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
    ) -> Dataset:
        if batch_size is None:
            future = function.apply_app(
                arg=None,
                outputs_=outputs,
                inputs=[self.extxyz],
                outputs=[psiflow.context().new_file('data_', '.xyz')],
                **function.parameters(),
            )
        else:
            future = batch_apply(
                function.apply_app,
                self,
                batch_size,
                self.length(),
                outputs=[psiflow.context().new_file('data_', '.xyz')],
                func_outputs=outputs,
                **function.parameters(),
            )
        return Dataset(None, extxyz=future.outputs[0])


    @classmethod
    def load(
        cls,
        path_xyz: Union[Path, str],
    ) -> Dataset:
        path_xyz = resolve_and_check(Path(path_xyz))
        assert path_xyz.exists()  # needs to be locally accessible
        return cls(None, extxyz=File(str(path_xyz)))


@typeguard.typechecked
def _write_frames(
    *states: Geometry,
    extra_states: Optional[list[Geometry]] = None,
    outputs: list = [],
) -> None:
    all_states = list(states)
    if extra_states is not None:
        all_states += extra_states
    with open(outputs[0], "w") as f:
        for state in all_states:
            f.write(state.to_string() + "\n")


write_frames = python_app(_write_frames, executors=["default_threads"])


@typeguard.typechecked
def _read_frames(
    indices: Union[None, slice, list[int], int] = None,
    inputs: list = [],
    outputs: list = [],
) -> Optional[list[Geometry]]:
    frame_index = 0
    frame_regex = re.compile(r"^\d+$")
    length = _count_frames(inputs=inputs)
    _all = range(length)
    if isinstance(indices, slice):
        indices = list(_all[indices])
    elif isinstance(indices, int):
        indices = [list(_all)[indices]]

    if isinstance(indices, list):
        if length > 0:
            indices = [i % length for i in indices]  # for negative indices and wrapping
        indices_ = set(indices)  # for *much* faster 'i in indices'
    else:
        assert indices is None
        indices_ = None

    data = []
    with open(inputs[0], "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if frame_regex.match(line.strip()):
                natoms = int(line.strip())

                # currently at position frame_index, check if to be read
                _ = [f.readline() for _i in range(natoms + 1)]
                if indices_ is None or frame_index in indices_:
                    data.append("".join([line] + _))
                else:
                    data.append(None)
                frame_index += 1

    if indices is not None:  # sort states accordingly
        data = [data[i] for i in indices]

    if len(outputs) > 0:
        with open(outputs[0], "w") as f:
            f.write("\n".join([d for d in data if d is not None]))
    else:
        geometries = [Geometry.from_string(s) for s in data if s is not None]
        return geometries


read_frames = python_app(_read_frames, executors=["default_threads"])


@typeguard.typechecked
def _check_equality(
    state0: Geometry,
    state1: Geometry,
) -> bool:
    return state0 == state1


check_equality = python_app(_check_equality, executors=["default_threads"])


@typeguard.typechecked
def _extract_quantities(
    quantities: tuple[str, ...],
    atom_indices: Optional[list[int]],
    elements: Optional[list[str]],
    data: Optional[list[Geometry]] = None,
    inputs: list = [],
) -> tuple[np.ndarray, ...]:
    if data is None:
        assert len(inputs) == 1
        data = _read_frames(inputs=inputs)
    else:
        assert len(inputs) == 0
    order_names = list(set([k for g in data for k in g.order]))
    natoms = np.array([len(geometry) for geometry in data], dtype=int)
    max_natoms = np.max(natoms)

    arrays = create_outputs(quantities, data)
    for i, geometry in enumerate(data):
        mask = get_index_element_mask(
            geometry.per_atom.numbers,
            atom_indices,
            elements,
            natoms_padded=int(max_natoms),
        )
        natoms = len(geometry)
        for j, quantity in enumerate(quantities):
            if quantity == "positions":
                arrays[j][i, mask, :] = geometry.per_atom.positions[mask[:natoms]]
            elif quantity == "forces":
                arrays[j][i, mask, :] = geometry.per_atom.forces[mask[:natoms]]
            elif quantity == "cell":
                arrays[j][i, :, :] = geometry.cell
            elif quantity == "stress":
                if geometry.stress is not None:
                    arrays[j][i, :, :] = geometry.stress
            elif quantity == "numbers":
                arrays[j][i, :] = geometry.numbers
            elif quantity == "energy":
                if geometry.energy is not None:
                    arrays[j][i] = geometry.energy
            elif quantity == "delta":
                if geometry.delta is not None:
                    arrays[j][i] = geometry.delta
            elif quantity == "per_atom_energy":
                if geometry.energy is not None:
                    arrays[j][i] = geometry.per_atom_energy
            elif quantity == "phase":
                if geometry.phase is not None:
                    arrays[j][i] = geometry.phase
            elif quantity == "logprob":
                if geometry.logprob is not None:
                    arrays[j][i, :] = geometry.logprob
            elif quantity == "identifier":
                if geometry.identifier is not None:
                    arrays[j][i] = geometry.identifier
            elif quantity in order_names:
                if quantity in geometry.order:
                    arrays[j][i] = geometry.order[quantity]
    return tuple(arrays)


extract_quantities = python_app(_extract_quantities, executors=["default_threads"])


@typeguard.typechecked
def _check_distances(state: Geometry, threshold: float) -> Geometry:
    from ase.geometry.geometry import find_mic

    if state == NullState:
        return NullState
    nrows = int(len(state) * (len(state) - 1) / 2)
    deltas = np.zeros((nrows, 3))
    count = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            deltas[count] = state.per_atom.positions[i] - state.per_atom.positions[j]
            count += 1
    assert count == nrows
    if state.periodic:
        deltas, _ = find_mic(deltas, state.cell)
    check = np.all(np.linalg.norm(deltas, axis=1) > threshold)
    if check:
        return state
    else:
        return NullState


check_distances = python_app(_check_distances, executors=["default_htex"])


def _assign_identifier(
    state: Geometry,
    identifier: int,
    discard: bool = False,
) -> tuple[Geometry, int]:
    if (state == NullState) or discard:
        return state, identifier
    else:
        assert state.identifier is None
        state.identifier = identifier
        return state, identifier + 1


assign_identifier = python_app(_assign_identifier, executors=["default_threads"])


@typeguard.typechecked
def _assign_identifiers(
    identifier: Optional[int],
    inputs: list = [],
    outputs: list = [],
) -> int:
    data = _read_frames(slice(None), inputs=[inputs[0]])
    states = []
    if identifier is None:  # do not assign but look for max
        identifier = -1
        for geometry in data:
            if geometry.identifier is not None:
                identifier = max(identifier, geometry.identifier)
        identifier += 1
        for geometry in data:  # assign those which were not yet assigned
            if geometry.identifier is None:
                geometry, identifier = _assign_identifier(geometry, identifier)
            states.append(geometry)
        _write_frames(*states, outputs=[outputs[0]])
        return identifier
    else:
        for geometry in data:
            geometry, identifier = _assign_identifier(geometry, identifier)
            states.append(geometry)
        _write_frames(*states, outputs=[outputs[0]])
        return identifier


assign_identifiers = python_app(_assign_identifiers, executors=["default_threads"])


@typeguard.typechecked
def _join_frames(
    inputs: list = [],
    outputs: list = [],
):
    assert len(outputs) == 1

    with open(outputs[0], "wb") as destination:
        for input_file in inputs:
            with open(input_file, "rb") as source:
                shutil.copyfileobj(source, destination)


join_frames = python_app(_join_frames, executors=["default_threads"])


@typeguard.typechecked
def _count_frames(inputs: list = []) -> int:
    nframes = 0
    frame_regex = re.compile(r"^\d+$")
    with open(inputs[0], "r") as f:
        for line in f:
            if frame_regex.match(line.strip()):
                nframes += 1
                natoms = int(line.strip())
                _ = [f.readline() for _i in range(natoms + 1)]
    return nframes


count_frames = python_app(_count_frames, executors=["default_threads"])


@typeguard.typechecked
def _reset_frames(inputs: list = [], outputs: list = []) -> None:
    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        geometry.reset()
    _write_frames(*data, outputs=[outputs[0]])


reset_frames = python_app(_reset_frames, executors=["default_threads"])


@typeguard.typechecked
def _clean_frames(inputs: list = [], outputs: list = []) -> None:
    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        geometry.clean()
    _write_frames(*data, outputs=[outputs[0]])


clean_frames = python_app(_clean_frames, executors=["default_threads"])


@typeguard.typechecked
def _apply_offset(
    subtract: bool,
    inputs: list = [],
    outputs: list = [],
    **atomic_energies: float,
) -> None:
    assert len(inputs) == 1
    assert len(outputs) == 1
    data = _read_frames(inputs=[inputs[0]])
    numbers = [atomic_numbers[e] for e in atomic_energies.keys()]
    all_numbers = [n for geometry in data for n in set(geometry.per_atom.numbers)]
    for n in all_numbers:
        if n != 0:  # from NullState
            assert n in numbers
    for geometry in data:
        if geometry == NullState:
            continue
        natoms = len(geometry)
        energy = geometry.energy
        for number in numbers:
            natoms_per_number = np.sum(geometry.per_atom.numbers == number)
            if natoms_per_number == 0:
                continue
            element = chemical_symbols[number]
            multiplier = -1 if subtract else 1
            energy += multiplier * natoms_per_number * atomic_energies[element]
            natoms -= natoms_per_number
        assert natoms == 0  # all atoms accounted for
        geometry.energy = energy
    _write_frames(*data, outputs=[outputs[0]])


apply_offset = python_app(_apply_offset, executors=["default_threads"])


@typeguard.typechecked
def _get_elements(inputs: list = []) -> set[str]:
    data = _read_frames(inputs=[inputs[0]])
    return set([chemical_symbols[n] for g in data for n in g.per_atom.numbers])


get_elements = python_app(_get_elements, executors=["default_threads"])


@typeguard.typechecked
def _align_axes(inputs: list = [], outputs: list = []) -> None:
    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        geometry.align_axes()
    _write_frames(*data, outputs=[outputs[0]])


align_axes = python_app(_align_axes, executors=["default_threads"])


@typeguard.typechecked
def _not_null(inputs: list = [], outputs: list = []) -> list[bool]:
    frame_regex = re.compile(r"^\d+$")

    data = []
    mask = []
    with open(inputs[0], "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if frame_regex.match(line.strip()):
                natoms = int(line.strip())
                _ = [f.readline() for _i in range(natoms + 1)]
                if natoms == 1:
                    if _[-1].strip()[0] == "X":  # check if element(-1) == X
                        mask.append(False)
                        continue
                data.append("".join([line] + _))
                mask.append(True)
    if len(outputs) > 0:
        with open(outputs[0], "w") as f:
            f.write("\n".join(data))
    return mask


not_null = python_app(_not_null, executors=["default_threads"])


@typeguard.typechecked
def _app_filter(
    quantity: str,
    inputs: list = [],
    outputs: list = [],
) -> None:
    data = _read_frames(inputs=[inputs[0]])
    i = 0
    while i < len(data):
        if quantity == "forces":
            if np.all(np.invert(np.isnan(data[i].per_atom.forces))):
                retain = True
            else:
                retain = False
        elif quantity == "cell":
            if not np.allclose(data[i].cell, 0.0):
                retain = True
            else:
                retain = False
        elif hasattr(data[i], quantity) and getattr(data[i], quantity) is not None:
            retain = True
        else:
            retain = False

        # pop if necessary:
        if retain:
            i += 1
        else:
            data.pop(i)

    _write_frames(*data, outputs=[outputs[0]])


app_filter = python_app(
    _app_filter, executors=["default_threads"]
)  # filter is protected


@typeguard.typechecked
def _shuffle(
    inputs: list = [],
    outputs: list = [],
) -> None:
    data = _read_frames(inputs=[inputs[0]])
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    shuffled = [data[int(i)] for i in indices]
    _write_frames(*shuffled, outputs=[outputs[0]])


shuffle = python_app(_shuffle, executors=["default_threads"])


@typeguard.typechecked
def _train_valid_indices(
    effective_nstates: int,
    train_valid_split: float,
    shuffle: bool,
) -> tuple[list[int], list[int]]:
    ntrain = int(np.floor(effective_nstates * train_valid_split))
    nvalid = effective_nstates - ntrain
    assert ntrain > 0
    assert nvalid > 0
    order = np.arange(ntrain + nvalid, dtype=int)
    if shuffle:
        np.random.shuffle(order)
    train_list = list(order[:ntrain])
    valid_list = list(order[ntrain : (ntrain + nvalid)])
    return [int(i) for i in train_list], [int(i) for i in valid_list]


train_valid_indices = python_app(_train_valid_indices, executors=["default_threads"])


@typeguard.typechecked
def get_train_valid_indices(
    effective_nstates: AppFuture,
    train_valid_split: float,
    shuffle: bool,
) -> tuple[AppFuture, AppFuture]:
    future = train_valid_indices(effective_nstates, train_valid_split, shuffle)
    return unpack_i(future, 0), unpack_i(future, 1)


@typeguard.typechecked
def get_index_element_mask(
    numbers: np.ndarray,
    atom_indices: Optional[list[int]],
    elements: Optional[list[str]],
    natoms_padded: Optional[int] = None,
) -> np.ndarray:
    mask = np.array([True] * len(numbers))

    if elements is not None:
        numbers_to_include = [atomic_numbers[e] for e in elements]
        mask_elements = np.array([False] * len(numbers))
        for number in numbers_to_include:
            mask_elements = np.logical_or(mask_elements, (numbers == number))
        mask = np.logical_and(mask, mask_elements)

    if natoms_padded is not None:
        assert natoms_padded >= len(numbers)
        padding = natoms_padded - len(numbers)
        mask = np.concatenate((mask, np.array([False] * padding)), axis=0).astype(bool)

    if atom_indices is not None:  # below padding
        mask_indices = np.array([False] * len(mask))
        mask_indices[np.array(atom_indices)] = True
        mask = np.logical_and(mask, mask_indices)

    return mask


@typeguard.typechecked
def _compute_rmse(
    array0: np.ndarray,
    array1: np.ndarray,
    reduce: bool = True,
) -> Union[float, np.ndarray]:
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


@typeguard.typechecked
def _batch_frames(
    batch_size: int,
    inputs: list = [],
    outputs: list = [],
) -> Optional[list[Geometry]]:
    frame_regex = re.compile(r"^\d+$")

    data = []
    batch_index = 0
    with open(inputs[0], "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if frame_regex.match(line.strip()):
                natoms = int(line.strip())
                _ = [f.readline() for _i in range(natoms + 1)]
                data.append("".join([line] + _))
                if len(data) == batch_size:
                    with open(outputs[batch_index], "w") as g:
                        g.write("\n".join(data))
                    data = []
                    batch_index += 1
                else:  # write and clear
                    pass
    if len(data) > 0:
        with open(outputs[batch_index], "w") as g:
            g.write("\n".join(data))
        batch_index += 1
    assert batch_index == len(outputs)


batch_frames = python_app(_batch_frames, executors=["default_threads"])


@join_app
@typeguard.typechecked
def batch_apply(
    apply_app: PythonApp,
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

    if reduce_func is None:
        assert len(outputs) == 1
    else:
        assert len(outputs) == 0

    output_futures = []
    for i in range(nbatches):
        f = apply_app(
            None,
            inputs=[future.outputs[i]],
            outputs=[batches[i]],  # has to be File, not DataFuture
            **app_kwargs,
        )
        output_futures.append(f)
    output_batches = [f.outputs[0] for f in output_futures]

    if reduce_func is None:
        f = join_frames(inputs=output_batches, outputs=[outputs[0]])
    else:
        assert reduce_func is not None
        f = reduce_func(*output_futures)
    return f
