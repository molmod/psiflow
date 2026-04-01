import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union, Sequence, Generator, TypeAlias

import numpy as np
import typeguard
from ase.data import atomic_numbers, chemical_symbols
from parsl import python_app, File
from parsl.dataflow.futures import AppFuture

from psiflow.geometry import Geometry, get_atomic_energy, get_unique_numbers

# TODO: Note: This function is wrapped as a Parsl app and executed using the default_threads executor.
# TODO: loosen type hints?
# TODO: inputs/outputs for apps sometimes behave weird


FileLike: TypeAlias = str | Path | File


class MissingType:
    """Placeholder sentinel for missing data fields"""

    def __repr__(self):
        return "<MISSING>"

    def __bool__(self):
        return False


MISSING = MissingType()


def iter_read_frames(file: FileLike) -> Generator[list[str]]:
    """Yields text data to instantiate geometries"""
    frame_regex = re.compile(r"^\d+$")
    with open(file, "r") as f:
        for line in f:
            if frame_regex.match(_ := line.strip()):
                n = int(_)
                yield [line] + [f.readline() for _i in range(n + 1)]


def _write_frames(
    *states: Geometry | list[Geometry], outputs: Sequence[File] = ()
) -> None:
    """
    Write Geometry instances to a file.

    Args:
        states: Variable number of (lists of) Geometry instances to write.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path.
    """
    assert len(outputs) == 1
    data = []
    for d in states:
        if isinstance(d, list):
            data.extend(d)
        else:
            data.append(d)
    with open(outputs[0], "w") as f:
        f.write("".join([geom.to_string() for geom in data]))


write_frames = python_app(_write_frames, executors=["default_threads"])


def _read_frames(
    file: FileLike, indices: Optional[slice | list[int] | int] = None
) -> list[Geometry]:
    """
    Read Geometry instances from a file.

    Args:
        file: DataFuture representing the input file path containing the geometry data.
        indices: Indices of frames to read. Can be None (read all), a slice, a list of integers, or a single integer.

    """
    if indices is None:
        # read everything
        data = ["".join(geom_str_list) for geom_str_list in iter_read_frames(file)]
        return [Geometry.from_string(s) for s in data]

    # figure out what frames to read
    # TODO: reads twice + indices always wrap modulo nframes which might be unexpected
    length = _count_frames(file)
    if isinstance(indices, slice):
        indices = list(range(length)[indices])
    elif isinstance(indices, int):
        indices = [indices]

    if length > 0:
        indices = [i % length for i in indices]  # for negative indices and wrapping
    indices_ = set(indices)  # for *much* faster 'i in indices'

    data = {}
    for i, geom_str_list in enumerate(iter_read_frames(file)):
        if i in indices_:
            data[i] = Geometry.from_string("".join(geom_str_list))

    # return in original order
    return [data[i] for i in indices]


read_frames = python_app(_read_frames, executors=["default_threads"])


def _extract_quantities(
    states: Sequence[Geometry], quantities: Sequence[str]
) -> dict[str, list]:
    """
    Extract specified quantities from Geometry instances.
    """
    data = {k: [] for k in quantities}
    for k in quantities:
        if k in ("cell", "energy", "stress"):
            for geom in states:
                value = getattr(geom, k, MISSING)
                data[k].append(value)
            continue
        for geom in states:
            value = getattr(geom.per_atom, k, MISSING)
            if value is MISSING:
                value = getattr(geom.meta, k, MISSING)
            data[k].append(value)
    return data


extract_quantities = python_app(_extract_quantities, executors=["default_threads"])


def _insert_quantities(
    states: Sequence[Geometry], data: dict[str, list]
) -> Sequence[Geometry]:
    """
    Insert quantities from data into Geometry instances.
    """
    for q, values in data.items():
        assert len(states) == len(values)

        if q in ("cell", "energy", "stress"):
            for geom, v in zip(states, values):
                setattr(geom, q, v)
            continue

        for geom, v in zip(states, values):
            if isinstance(v, np.ndarray) and len(geom) == len(v):
                setattr(geom.per_atom, q, v)
            else:
                setattr(geom.meta, q, v)

    return states


insert_quantities = python_app(_insert_quantities, executors=["default_threads"])


def _extract_quantities_per_atom(
    states: Sequence[Geometry],
    quantities: Sequence[str],
    atom_indices: Optional[Sequence[int]] = None,
    elements: Optional[Sequence[str]] = None,
) -> dict[str, list]:
    """
    Extract per atom quantities from Geometry instances, filtering based on atom_indices or elements.
    """
    if atom_indices is None and elements is None:
        # no filtering
        data = {}
        for k in quantities:
            data[k] = [getattr(geom.per_atom, k, MISSING) for geom in states]
        return data

    data = {k: [] for k in quantities}
    numbers = {atomic_numbers[s] for s in elements or ()}
    for geom in states:
        # mask out unwanted rows
        mask = np.zeros(len(geom), dtype=bool)
        for n in numbers:
            mask[geom.numbers == n] = True
        if atom_indices is not None:
            mask[atom_indices] = True

        for k in quantities:
            value = getattr(geom.per_atom, k, MISSING)
            if not value is MISSING:
                value = value[mask]
            data[k].append(value)

    return data


extract_quantities_per_atom = python_app(
    _extract_quantities_per_atom, executors=["default_threads"]
)


@typeguard.typechecked
def _check_distances(state: Geometry, threshold: float) -> Geometry:
    """
    Check if all interatomic distances in a Geometry are above a threshold.

    Args:
        state: Geometry instance to check.
        threshold: Minimum allowed interatomic distance.

    Returns:
        Geometry: The input Geometry if all distances are above the threshold, otherwise NullState.

    Note:
        This function is wrapped as a Parsl app and executed using the default_htex executor.
    """
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


@typeguard.typechecked
def _assign_identifiers(
    identifier: Optional[int],
    inputs: list = [],
    outputs: list = [],
) -> int:
    """
    Assign identifiers to Geometry instances in a file.

    Args:
        identifier: Starting identifier value.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where updated geometries will be written.

    Returns:
        int: Next available identifier.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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


def _join_frames(inputs: Sequence[File] = (), outputs: Sequence[File] = ()) -> None:
    """
    Join multiple frame files into a single file.

    Args:
        inputs: List of Parsl futures. Each element should be a DataFuture
                representing an input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where joined frames will be written.
    """
    assert len(outputs) == 1

    with open(outputs[0], "wb") as destination:
        for input_file in inputs:
            with open(input_file, "rb") as source:
                shutil.copyfileobj(source, destination)


join_frames = python_app(_join_frames, executors=["default_threads"])


def _count_frames(file: FileLike) -> int:
    """Count the number of frames in a file."""
    path = Path(file)
    threshold = 10 * 1024 * 1024  # 10 MB in bytes
    file_size = path.stat().st_size

    if file_size > threshold:  # use grep
        cmd = f"grep -Fc Properties {path}"
        result = subprocess.check_output(cmd.split())
        return int(result.strip())
    return len([_ for _ in iter_read_frames(path)])


count_frames = python_app(_count_frames, executors=["default_threads"])


def _reset_frames(file: FileLike, outputs: Sequence[File] = ()) -> None:
    """
    Reset all frames in a file.

    Args:
        file: DataFuture representing the input file path containing the geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where reset frames will be written.
    """
    assert len(outputs) == 1
    data = _read_frames(file)
    for geometry in data:
        geometry.reset()
    _write_frames(*data, outputs=outputs)


reset_frames = python_app(_reset_frames, executors=["default_threads"])


def _clean_frames(file: FileLike, outputs: Sequence[File] = ()) -> None:
    """
    Clean all frames in a file.

    Args:
        file: DataFuture representing the input file path containing the geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where reset frames will be written.
    """
    assert len(outputs) == 1
    data = _read_frames(file)
    for geometry in data:
        geometry.clean()
    _write_frames(*data, outputs=outputs)


clean_frames = python_app(_clean_frames, executors=["default_threads"])


def _apply_offset(
    file: FileLike,
    subtract: bool,
    outputs: Sequence[File] = (),
    **atomic_energies: float,
) -> None:
    """
    Apply an energy offset to all frames in a file.

    Args:
        file: DataFuture representing the input file path containing the geometry data.
        subtract: Whether to subtract or add the offset.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where updated frames will be written.
        **atomic_energies: Atomic energies for each element.
    """
    assert len(outputs) == 1
    frames = _read_frames(file)
    numbers_data = get_unique_numbers(frames)
    numbers_kwargs = {atomic_numbers[e] for e in atomic_energies.keys()}
    assert numbers_data == numbers_kwargs, "Provide atomic energies for all elements.."

    for geom in frames:
        energy = get_atomic_energy(geom, atomic_energies)
        if subtract:
            geom.energy -= energy
        else:
            geom.energy += energy

    _write_frames(frames, outputs=outputs)


apply_offset = python_app(_apply_offset, executors=["default_threads"])


def _get_elements(*files: File) -> set[str]:
    """
    Get the set of elements present in all frames of a sequence of file.

    Args:
        inputs: List of Parsl DataFuture
    """
    frames = [geom for file in files for geom in _read_frames(file)]
    return {chemical_symbols[i] for i in get_unique_numbers(frames)}


get_elements = python_app(_get_elements, executors=["default_threads"])


def _align_axes(file: FileLike, outputs: Sequence[File] = ()) -> None:
    """
    Align axes for all frames in a file.

    Args:
        file: DataFuture representing the input file path containing the geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where aligned frames will be written.
    """
    assert len(outputs) == 1
    data = _read_frames(file)
    for geometry in data:
        geometry.align_axes()
    _write_frames(*data, outputs=outputs)


align_axes = python_app(_align_axes, executors=["default_threads"])


def _app_filter(file: FileLike, quantity: str, outputs: Sequence[File] = ()) -> None:
    """
    Filter frames based on a specified quantity and writes to first output.
    """
    # TODO: where is this used?
    data = _read_frames(file)
    out = []

    # None does not count
    for geom in data:
        if (quantity in ("cell", "energy", "stress")) and getattr(
            geom, quantity, None
        ) is not None:
            out.append(geom)
            continue

        value = getattr(geom.per_atom, quantity, None)
        if value is None:
            value = getattr(geom.meta, quantity, None)
        if value is not None:
            out.append(geom)

    _write_frames(out, outputs=outputs)


# filter is protected
app_filter = python_app(_app_filter, executors=["default_threads"])


def _shuffle(
    file: FileLike,
    outputs: Sequence[File] = (),
) -> None:
    """
    Shuffle the order of frames in a file.

    Args:
        file: DataFuture representing the input file path containing the geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where shuffled frames will be written.
    """
    assert len(outputs) == 1
    frames = _read_frames(file)
    np.random.shuffle(frames)
    _write_frames(frames, outputs=outputs)


shuffle = python_app(_shuffle, executors=["default_threads"])


@typeguard.typechecked
def _train_valid_indices(
    effective_nstates: int,
    train_valid_split: float,
    shuffle: bool,
) -> tuple[list[int], list[int]]:
    """
    Generate indices for train and validation splits.

    Args:
        effective_nstates: Total number of states.
        train_valid_split: Fraction of states to use for training.
        shuffle: Whether to shuffle the indices.

    Returns:
        tuple[list[int], list[int]]: Lists of indices for training and validation sets.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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
    """
    Get futures for train and validation indices.

    Args:
        effective_nstates: Future representing the total number of states.
        train_valid_split: Fraction of states to use for training.
        shuffle: Whether to shuffle the indices.

    Returns:
        tuple[AppFuture, AppFuture]: Futures for training and validation indices.
    """
    future = train_valid_indices(effective_nstates, train_valid_split, shuffle)
    return future[0], future[1]


# @typeguard.typechecked
# def get_index_element_mask(
#     numbers: np.ndarray,
#     atom_indices: Optional[list[int]],
#     elements: Optional[list[str]],
#     natoms_padded: Optional[int] = None,
# ) -> np.ndarray:
#     """
#     Generate a mask for atom indices and elements.
#
#     Args:
#         numbers: Array of atomic numbers.
#         atom_indices: List of atom indices to include.
#         elements: List of element symbols to include.
#         natoms_padded: Total number of atoms including padding.
#
#     Returns:
#         np.ndarray: Boolean mask array.
#     """
#     mask = np.array([True] * len(numbers))
#
#     if elements is not None:
#         numbers_to_include = [atomic_numbers[e] for e in elements]
#         mask_elements = np.array([False] * len(numbers))
#         for number in numbers_to_include:
#             mask_elements = np.logical_or(mask_elements, (numbers == number))
#         mask = np.logical_and(mask, mask_elements)
#
#     if natoms_padded is not None:
#         assert natoms_padded >= len(numbers)
#         padding = natoms_padded - len(numbers)
#         mask = np.concatenate((mask, np.array([False] * padding)), axis=0).astype(bool)
#
#     if atom_indices is not None:  # below padding
#         mask_indices = np.array([False] * len(mask))
#         mask_indices[np.array(atom_indices)] = True
#         mask = np.logical_and(mask, mask_indices)
#
#     return mask

# TODO: these do not belong here

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


@typeguard.typechecked
def _batch_frames(
    batch_size: int,
    inputs: list = [],
    outputs: list = [],
) -> Optional[list[Geometry]]:
    """
    Split frames into batches.

    Args:
        batch_size: Number of frames per batch.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. Each element should be a DataFuture
                 representing an output file path for each batch.

    Returns:
        Optional[list[Geometry]]: List of Geometry instances if no outputs are specified.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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
        with open(outputs[batch_index], "w") as f:
            f.write("\n".join([d.strip() for d in data if d is not None]))
            f.write("\n")
        batch_index += 1
    assert batch_index == len(outputs)


batch_frames = python_app(_batch_frames, executors=["default_threads"])
