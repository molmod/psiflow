from typing import Optional

import numpy as np
import typeguard
from ase.data import atomic_numbers
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

from psiflow.utils import unpack_i


def is_lower_triangular(cell: np.ndarray) -> bool:
    return (
        cell[0, 0] > 0
        and cell[1, 1] > 0  # positive volumes
        and cell[2, 2] > 0
        and cell[0, 1] == 0
        and cell[0, 2] == 0  # lower triangular
        and cell[1, 2] == 0
    )


def is_reduced(cell: np.ndarray) -> bool:
    return (
        cell[0, 0] > abs(2 * cell[1, 0])
        and cell[0, 0] > abs(2 * cell[2, 0])  # b mostly along y axis
        and cell[1, 1] > abs(2 * cell[2, 1])  # c mostly along z axis
        and is_lower_triangular(cell)  # c mostly along z axis
    )


def transform_lower_triangular(
    pos: np.ndarray, cell: np.ndarray, reorder: bool = False
):
    """Transforms coordinate axes such that cell matrix is lower diagonal

    The transformation is derived from the QR decomposition and performed
    in-place. Because the lower triangular form puts restrictions on the size
    of off-diagonal elements, lattice vectors are by default reordered from
    largest to smallest; this feature can be disabled using the reorder
    keyword.
    The box vector lengths and angles remain exactly the same.

    """
    if reorder:  # reorder box vectors as k, l, m with |k| >= |l| >= |m|
        norms = np.linalg.norm(cell, axis=1)
        ordering = np.argsort(norms)[::-1]  # largest first
        a = cell[ordering[0], :].copy()
        b = cell[ordering[1], :].copy()
        c = cell[ordering[2], :].copy()
        cell[0, :] = a[:]
        cell[1, :] = b[:]
        cell[2, :] = c[:]
    q, r = np.linalg.qr(cell.T)
    flip_vectors = np.eye(3) * np.diag(np.sign(r))  # reflections after rotation
    rotation = np.linalg.inv(q.T) @ flip_vectors  # full (improper) rotation
    pos[:] = pos @ rotation
    cell[:] = cell @ rotation
    assert np.allclose(cell, np.linalg.cholesky(cell @ cell.T), atol=1e-5)
    cell[0, 1] = 0
    cell[0, 2] = 0
    cell[1, 2] = 0


def reduce_box_vectors(cell: np.ndarray):
    """Uses linear combinations of box vectors to obtain the reduced form

    The reduced form of a cell matrix is lower triangular, with additional
    constraints that enforce vector b to lie mostly along the y-axis and vector
    c to lie mostly along the z axis.

    """
    # simple reduction algorithm only works on lower triangular cell matrices
    assert is_lower_triangular(cell)
    # replace c and b with shortest possible vectors to ensure
    # b_y > |2 c_y|
    # b_x > |2 c_x|
    # a_x > |2 b_x|
    cell[2, :] = cell[2, :] - cell[1, :] * np.round(cell[2, 1] / cell[1, 1])
    cell[2, :] = cell[2, :] - cell[0, :] * np.round(cell[2, 0] / cell[0, 0])
    cell[1, :] = cell[1, :] - cell[0, :] * np.round(cell[1, 0] / cell[0, 0])


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
    from psiflow.data.geometry import _read_frames, _write_frames

    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        geometry.reset()
    _write_frames(*data, outputs=[outputs[0]])


reset_frames = python_app(_reset_frames, executors=["default_threads"])


@typeguard.typechecked
def _clean_frames(inputs: list = [], outputs: list = []) -> None:
    from psiflow.data.geometry import _read_frames, _write_frames

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
    from ase.data import atomic_numbers, chemical_symbols

    from psiflow.data.geometry import NullState, _read_frames, _write_frames

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
    from ase.data import chemical_symbols

    from psiflow.data.geometry import _read_frames

    data = _read_frames(inputs=[inputs[0]])
    return set([chemical_symbols[n] for g in data for n in g.per_atom.numbers])


get_elements = python_app(_get_elements, executors=["default_threads"])


@typeguard.typechecked
def _align_axes(inputs: list = [], outputs: list = []) -> None:
    from psiflow.data.geometry import _read_frames, _write_frames

    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        geometry.align_axes()
    _write_frames(*data, outputs=[outputs[0]])


align_axes = python_app(_align_axes, executors=["default_threads"])


@typeguard.typechecked
def _not_null(inputs: list = [], outputs: list = []) -> None:
    from psiflow.data.geometry import NullState, _read_frames, _write_frames

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
    elements: Optional[list[str]],
    atom_indices: Optional[list[int]],
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
    return mask
