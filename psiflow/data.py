from __future__ import annotations  # necessary for type-guarding class methods

from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from ase.data import atomic_numbers, chemical_symbols
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.geometry import Geometry, NullState
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


@typeguard.typechecked
def _write_frames(*states: Geometry, outputs: list = []) -> None:
    with open(outputs[0], "w") as f:
        for state in states:
            f.write(state.to_string() + "\n")


write_frames = python_app(_write_frames, executors=["default_threads"])


@typeguard.typechecked
def _read_frames(
    indices: Union[None, slice, list[int], int] = None,
    # safe: bool = False,
    inputs: list = [],
    outputs: list = [],
) -> Optional[list[Geometry]]:
    length = _count_frames(inputs=inputs)
    if isinstance(indices, slice):
        indices = list(range(length)[indices])
    elif isinstance(indices, int):
        indices = [indices]
    elif indices is None:
        indices = list(range(length))

    # should have converted everything to list
    assert type(indices) is list
    indices = [i % length for i in indices]

    data = []
    frame_count = 0
    with open(inputs[0], "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            natoms = int(line)
            if (indices is None) or (frame_count in indices):
                lines = [f.readline() for i in range(natoms + 1)]
                data.append(Geometry.from_string("".join(lines), natoms))
            else:
                data.append(None)
                for _i in range(natoms + 1):  # skip ahead
                    f.readline()
            frame_count += 1
    assert frame_count == length

    if indices is not None:  # sort data according to indices!
        data = [data[i] for i in indices]

    if len(outputs) == 0:
        return data
    else:  # do not return data when it's written to file
        _write_frames(*data, outputs=[outputs[0]])
        return None


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
    QUANTITIES = [
        "positions",
        "cell",
        "numbers",
        "energy",
        "forces",
        "stress",
        "delta",
        "logprob",
        "phase",
        "identifier",
    ]
    assert all([q in QUANTITIES for q in quantities])
    if data is None:
        assert len(inputs) == 1
        data = _read_frames(inputs=inputs)
    else:
        assert len(inputs) == 0
    natoms = np.array([len(geometry) for geometry in data], dtype=int)
    max_natoms = np.max(natoms)
    nframes = len(data)
    nprob = 0
    for state in data:
        if state.logprob is not None:
            nprob = max(len(state.logprob), nprob)
    arrays = []
    for quantity in quantities:
        if quantity in ["positions", "forces"]:
            array = np.empty((nframes, max_natoms, 3), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["cell", "stress"]:
            array = np.empty((nframes, 3, 3), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["numbers"]:
            array = np.empty((nframes, max_natoms, 1), dtype=np.uint8)
            array[:] = 0
        elif quantity in ["energy", "delta", "per_atom_energy"]:
            array = np.empty((nframes,), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["phase"]:
            array = np.empty((nframes,), dtype=str)
            array[:] = ""
        elif quantity in ["logprob"]:
            array = np.empty((nframes, nprob, 1), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["identifier"]:
            array = np.empty((nframes,), dtype=np.int32)
            array[:] = -1
        else:
            raise AssertionError("missing quantity in if/else")
        arrays.append(array)

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
                arrays[j][i, :, :] = geometry.numbers
            elif quantity == "energy":
                if geometry.energy is not None:
                    arrays[j][i] = geometry.energy
            elif quantity == "delta":
                if geometry.delta is not None:
                    arrays[j][i] = geometry.delta
            elif quantity == "per_atom_energy":
                if geometry.energy is not None:
                    arrays[j][i] = geometry.energy / natoms[i]
            elif quantity == "phase":
                if geometry.phase is not None:
                    arrays[j][i] = geometry.phase
            elif quantity == "logprob":
                if geometry.logprob is not None:
                    arrays[j][i, :, :] = geometry.logprob
            elif quantity == "identifier":
                if geometry.identifier is not None:
                    arrays[j][i] = geometry.identifier
    return tuple(arrays)


extract_quantities = python_app(_extract_quantities, executors=["default_threads"])


@typeguard.typechecked
def _check_distances(state: Geometry, threshold: float) -> Geometry:
    import numpy as np
    from ase.geometry.geometry import find_mic

    from psiflow.data import NullState

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


check_distances = python_app(_check_distances, executors=["default_threads"])


@typeguard.typechecked
def _assign_identifier(state: Geometry, identifier: int):
    if not state == NullState:
        state.identifier = identifier
        identifier += 1
    return state, identifier


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
    import shutil

    assert len(outputs) == 1

    with open(outputs[0], "wb") as destination:
        for input_file in inputs:
            with open(input_file, "rb") as source:
                shutil.copyfileobj(source, destination)


join_frames = python_app(_join_frames, executors=["default_threads"])


@typeguard.typechecked
def _count_frames(inputs: list = []) -> int:
    nframes = 0
    with open(inputs[0], "r") as f:
        while True:
            try:
                natoms = int(f.readline())
            except ValueError:
                break
            nframes += 1
            for _i in range(natoms + 1):  # skip ahead
                f.readline()
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
    import numpy as np

    from psiflow.data import _read_frames, _write_frames
    from psiflow.geometry import NullState

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
    from psiflow.data import _read_frames

    data = _read_frames(inputs=[inputs[0]])
    return set([chemical_symbols[n] for g in data for n in g.per_atom.numbers])


get_elements = python_app(_get_elements, executors=["default_threads"])


@typeguard.typechecked
def _align_axes(inputs: list = [], outputs: list = []) -> None:
    from psiflow.data import _read_frames, _write_frames

    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        geometry.align_axes()
    _write_frames(*data, outputs=[outputs[0]])


align_axes = python_app(_align_axes, executors=["default_threads"])


@typeguard.typechecked
def _not_null(inputs: list = [], outputs: list = []) -> None:
    from psiflow.data import _read_frames, _write_frames
    from psiflow.geometry import NullState

    data = _read_frames(inputs=[inputs[0]])
    i = 0
    while i < len(data):
        if data[i] == NullState:
            data.pop(i)
        else:
            i += 1
    _write_frames(*data, outputs=[outputs[0]])


not_null = python_app(_not_null, executors=["default_threads"])


@typeguard.typechecked
def _train_valid_indices(
    effective_nstates: int,
    train_valid_split: float,
) -> tuple[list[int], list[int]]:
    import numpy as np

    ntrain = int(np.floor(effective_nstates * train_valid_split))
    nvalid = effective_nstates - ntrain
    assert ntrain > 0
    assert nvalid > 0
    order = np.arange(ntrain + nvalid, dtype=int)
    np.random.shuffle(order)
    train_list = list(order[:ntrain])
    valid_list = list(order[ntrain : (ntrain + nvalid)])
    return [int(i) for i in train_list], [int(i) for i in valid_list]


train_valid_indices = python_app(_train_valid_indices, executors=["default_threads"])


@typeguard.typechecked
def get_train_valid_indices(
    effective_nstates: AppFuture,
    train_valid_split: float,
) -> tuple[AppFuture, AppFuture]:
    future = train_valid_indices(effective_nstates, train_valid_split)
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

    if atom_indices is not None:
        mask_indices = np.array([False] * len(numbers))
        mask_indices[np.array(atom_indices)] = True
        mask = np.logical_and(mask, mask_indices)

    if natoms_padded is not None:
        assert natoms_padded >= len(numbers)
        padding = natoms_padded - len(numbers)
        mask = np.concatenate((mask, np.array([False] * padding)), axis=0).astype(bool)
    return mask


@typeguard.typechecked
def _compute_rmse(
    array0,
    array1,
) -> float:
    assert array0.shape == array1.shape
    mask0 = np.logical_not(np.isnan(array0))
    mask1 = np.logical_not(np.isnan(array1))
    assert np.all(mask0 == mask1)
    return np.sqrt(np.mean((array0[mask0] - array1[mask1]) ** 2))


compute_rmse = python_app(_compute_rmse, executors=["default_threads"])


@typeguard.typechecked
def _compute_mae(
    array0,
    array1,
) -> float:
    assert array0.shape == array1.shape
    mask0 = np.logical_not(np.isnan(array0))
    mask1 = np.logical_not(np.isnan(array1))
    assert np.all(mask0 == mask1)
    return np.mean(np.abs(array0[mask0] - array1[mask1]))


compute_mae = python_app(_compute_mae, executors=["default_threads"])
