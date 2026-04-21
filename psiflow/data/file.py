import re
import shutil
import subprocess
from pathlib import Path
from functools import partial
from typing import Optional, Generator, TypeAlias, Any, Callable
from collections.abc import Sequence

import numpy as np
from ase.data import chemical_symbols
from parsl import python_app, File

from psiflow.geometry import Geometry
from psiflow.data.utils import (
    get_unique_numbers,
    apply_energy_offset,
    filter_quantity,
    extract,
    extract_per_atom,
    assign_ids,
)


FileLike: TypeAlias = str | Path | File


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


def _get_elements(*files: File) -> set[str]:
    """
    Get the set of elements present in all frames of a sequence of file.
    """
    frames = [geom for file in files for geom in _read_frames(file)]
    return {chemical_symbols[i] for i in get_unique_numbers(frames)}


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


write_frames = python_app(_write_frames, executors=["default_threads"])
read_frames = python_app(_read_frames, executors=["default_threads"])
join_frames = python_app(_join_frames, executors=["default_threads"])
count_frames = python_app(_count_frames, executors=["default_threads"])
get_elements = python_app(_get_elements, executors=["default_threads"])
split_frames = python_app(_split_frames, executors=["default_threads"])


def _read_write_wrapper(
    execute: Callable,  # Parsl throws if this is named 'func'
    file: FileLike,
    *args: Any,
    outputs: Sequence[File] = (),
    **kwargs: Any,
) -> None:
    """
    Wrapper function to make simple Geometry functions work for Dataset instances.
    Essentially does (i) read (ii) modify (iii) write.

    Args:
        file: DataFuture representing the input file path containing the geometry data.
        execute: Function that will get executed for the geometries in file
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where reset frames will be written.
    """
    assert len(outputs) == 1
    frames_in = _read_frames(file)
    frames_out = execute(frames_in, *args, **kwargs)
    if frames_out is None:  # execute can alter in-place
        frames_out = frames_in
    _write_frames(frames_out, outputs=outputs)


def _reset_frames(states: Sequence[Geometry]) -> None:
    for geom in states:
        geom.reset()


def _clean_frames(states: Sequence[Geometry]) -> None:
    for geom in states:
        geom.clean()


def _align_axes(states: Sequence[Geometry]) -> None:
    for geom in states:
        geom.align_axes()


def _shuffle_frames(states: Sequence[Geometry]) -> None:
    np.random.shuffle(states)


read_write_app = python_app(_read_write_wrapper, executors=["default_threads"])
reset_frames = partial(read_write_app, _reset_frames)
clean_frames = partial(read_write_app, _clean_frames)
align_axes = partial(read_write_app, _align_axes)
shuffle_frames = partial(read_write_app, _shuffle_frames)

apply_offset = partial(read_write_app, apply_energy_offset)
filter_frames = partial(read_write_app, filter_quantity)


def _read_wrapper(
    execute: Callable,  # Parsl throws if this is named 'func'
    file: FileLike,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Wrapper function to make simple Geometry functions work for Dataset instances.
    Essentially does (i) read (ii) modify (iii) return.

    Args:
        file: DataFuture representing the input file path containing the geometry data.
        execute: Function that will get executed for the geometries in file
    """
    frames_in = _read_frames(file)
    return execute(frames_in, *args, **kwargs)


read_app = python_app(_read_wrapper, executors=["default_threads"])
extract_quantities = partial(read_app, extract)
extract_quantities_per_atom = partial(read_app, extract_per_atom)
assign_identifiers = partial(read_app, assign_ids)
