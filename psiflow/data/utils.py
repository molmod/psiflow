import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Generator, TypeAlias

import numpy as np
import typeguard
from ase.data import atomic_numbers, chemical_symbols
from parsl import python_app, File

from psiflow.geometry import Geometry, get_atomic_energy, get_unique_numbers

# TODO: Note: This function is wrapped as a Parsl app and executed using the default_threads executor.
# TODO: inputs/outputs for apps sometimes behave weird


FileLike: TypeAlias = str | Path | File


class MissingType:
    """Placeholder sentinel for missing data fields"""

    def __repr__(self):
        return "<MISSING>"

    def __bool__(self):
        return False


MISSING = MissingType()


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


def _split_frames(
    file: FileLike, fraction: float, shuffle: bool, outputs: Sequence[File] = ()
) -> None:
    """Split into training and validation sets"""
    assert len(outputs) == 2
    assert 0 <= fraction <= 1
    frames = _read_frames(file)
    order = np.arange(len(frames))
    if shuffle:
        np.random.shuffle(order)

    n_train = int(len(frames) * fraction)
    train_ids = order[:n_train]
    val_ids = order[n_train:]

    train = [frames[i] for i in train_ids]
    val = [frames[i] for i in val_ids]
    _write_frames(train, outputs=outputs[:1])
    _write_frames(val, outputs=outputs[1:])


split_frames = python_app(_split_frames, executors=["default_threads"])


# TODO
# TODO: think about where the parts below should live
# TODO


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


# def _assign_identifier(
#     state: Geometry,
#     identifier: int,
#     discard: bool = False,
# ) -> tuple[Geometry, int]:
#     """
#     Assign an identifier to a Geometry instance.
#
#     Args:
#         state (Geometry): Input Geometry instance.
#         identifier (int): Identifier to assign.
#         discard (bool, optional): Whether to discard the state. Defaults to False.
#
#     Returns:
#         tuple[Geometry, int]: Updated Geometry and next available identifier.
#     """
#     if (state == NullState) or discard:
#         return state, identifier
#     else:
#         assert state.identifier is None
#         state.identifier = identifier
#         return state, identifier + 1
#
#
# assign_identifier = python_app(_assign_identifier, executors=["default_threads"])
#


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
