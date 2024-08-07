import re
import shutil
from typing import Optional, Union

import numpy as np
import typeguard
from ase.data import atomic_numbers, chemical_symbols
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

from psiflow.geometry import Geometry, NullState, _assign_identifier, create_outputs
from psiflow.utils.apps import unpack_i


@typeguard.typechecked
def _write_frames(
    *states: Geometry,
    extra_states: Union[Geometry, list[Geometry], None] = None,
    outputs: list = [],
) -> None:
    """
    Write Geometry instances to a file.

    Args:
        *states: Variable number of Geometry instances to write.
        extra_states: Additional Geometry instance(s) to write.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    all_states = list(states)
    if extra_states is not None:
        if isinstance(extra_states, list):
            all_states += extra_states
        else:  # single geometry
            all_states.append(extra_states)
    with open(outputs[0], "w") as f:
        for state in all_states:  # avoid double newline by using strip!
            f.write(state.to_string().strip() + "\n")


write_frames = python_app(_write_frames, executors=["default_threads"])


@typeguard.typechecked
def _read_frames(
    indices: Union[None, slice, list[int], int] = None,
    inputs: list = [],
    outputs: list = [],
) -> Optional[list[Geometry]]:
    """
    Read Geometry instances from a file.

    Args:
        indices: Indices of frames to read. Can be None (read all), a slice, a list of integers, or a single integer.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing the geometry data.
        outputs: List of Parsl futures. If provided, the first element should be
                 a DataFuture representing the output file path where the selected
                 geometries will be written.

    Returns:
        Optional[list[Geometry]]: List of read Geometry instances if no output
                                  is specified, otherwise None.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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
            f.write("\n".join([d.strip() for d in data if d is not None]))
            f.write("\n")
    else:
        geometries = [Geometry.from_string(s) for s in data if s is not None]
        return geometries


read_frames = python_app(_read_frames, executors=["default_threads"])


@typeguard.typechecked
def _extract_quantities(
    quantities: tuple[str, ...],
    atom_indices: Optional[list[int]],
    elements: Optional[list[str]],
    *extra_data: Geometry,
    inputs: list = [],
) -> tuple[np.ndarray, ...]:
    """
    Extract specified quantities from Geometry instances.

    Args:
        quantities: Tuple of quantity names to extract.
        atom_indices: List of atom indices to consider.
        elements: List of element symbols to consider.
        *extra_data: Additional Geometry instances.
        inputs: List of Parsl futures. If provided, the first element should be a DataFuture
                representing the input file path containing geometry data.

    Returns:
        tuple[np.ndarray, ...]: Tuple of arrays containing extracted quantities.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    if not len(extra_data):
        assert len(inputs) == 1
        data = _read_frames(inputs=inputs)
    else:
        assert len(inputs) == 0
        data = list(extra_data)
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
def _insert_quantities(
    quantities: tuple[str, ...],
    arrays: list[np.ndarray, ...],
    data: Optional[list[Geometry]] = None,
    inputs: list = [],
    outputs: list = [],
) -> None:
    """
    Insert quantities into Geometry instances.

    Args:
        quantities: Tuple of quantity names to insert.
        arrays: List of arrays containing the quantities to insert.
        data: List of Geometry instances to update.
        inputs: List of Parsl futures. If provided, the first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. If provided, the first element should be a DataFuture
                 representing the output file path where updated geometries will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    if data is None:
        assert len(inputs) == 1
        data = _read_frames(inputs=inputs)
    else:
        assert len(inputs) == 0
    max_natoms = max([len(geometry) for geometry in data])

    for i, geometry in enumerate(data):
        if geometry == NullState:
            continue
        mask = get_index_element_mask(
            geometry.per_atom.numbers,
            None,
            None,
            natoms_padded=int(max_natoms),
        )
        natoms = len(geometry)
        for j, quantity in enumerate(quantities):
            if quantity == "positions":
                geometry.per_atom.positions[:natoms, :] = arrays[j][i, mask, :]
            elif quantity == "forces":
                geometry.per_atom.forces[mask[:natoms]] = arrays[j][i, mask, :]
            elif quantity == "cell":
                geometry.cell = arrays[j][i, :, :]
            elif quantity == "stress":
                geometry.stress = arrays[j][i, :, :]
            elif quantity == "numbers":
                geometry.numbers = arrays[j][i, :]
            elif quantity == "energy":
                geometry.energy = arrays[j][i]
            elif quantity == "delta":
                geometry.delta = arrays[j][i]
            elif quantity == "per_atom_energy":
                geometry.per_atom_energy = arrays[j][i]
            elif quantity == "phase":
                geometry.phase = arrays[j][i]
            elif quantity == "logprob":
                geometry.logprob = arrays[j][i, :]
            elif quantity == "identifier":
                geometry.identifier = arrays[j][i]
            elif quantity in geometry.order:
                geometry.order[quantity] = arrays[j][i]
            else:
                raise ValueError("unknown quantity {}".format(quantity))
    if len(outputs) > 0:
        _write_frames(*data, outputs=[outputs[0]])


insert_quantities = python_app(_insert_quantities, executors=["default_threads"])


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


@typeguard.typechecked
def _join_frames(
    inputs: list = [],
    outputs: list = [],
):
    """
    Join multiple frame files into a single file.

    Args:
        inputs: List of Parsl futures. Each element should be a DataFuture
                representing an input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where joined frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(outputs) == 1

    with open(outputs[0], "wb") as destination:
        for input_file in inputs:
            with open(input_file, "rb") as source:
                shutil.copyfileobj(source, destination)


join_frames = python_app(_join_frames, executors=["default_threads"])


@typeguard.typechecked
def _count_frames(inputs: list = []) -> int:
    """
    Count the number of frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.

    Returns:
        int: Number of frames in the file.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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
    """
    Reset all frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where reset frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        geometry.reset()
    _write_frames(*data, outputs=[outputs[0]])


reset_frames = python_app(_reset_frames, executors=["default_threads"])


@typeguard.typechecked
def _clean_frames(inputs: list = [], outputs: list = []) -> None:
    """
    Clean all frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where cleaned frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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
    """
    Apply an energy offset to all frames in a file.

    Args:
        subtract: Whether to subtract or add the offset.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where updated frames will be written.
        **atomic_energies: Atomic energies for each element.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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
    """
    Get the set of elements present in all frames of a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.

    Returns:
        set[str]: Set of element symbols.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    data = _read_frames(inputs=[inputs[0]])
    return set([chemical_symbols[n] for g in data for n in g.per_atom.numbers])


get_elements = python_app(_get_elements, executors=["default_threads"])


@typeguard.typechecked
def _align_axes(inputs: list = [], outputs: list = []) -> None:
    """
    Align axes for all frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where aligned frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        geometry.align_axes()
    _write_frames(*data, outputs=[outputs[0]])


align_axes = python_app(_align_axes, executors=["default_threads"])


@typeguard.typechecked
def _not_null(inputs: list = [], outputs: list = []) -> list[bool]:
    """
    Check which frames in a file are not null states.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. If provided, the first element should be a DataFuture
                 representing the output file path where non-null frames will be written.

    Returns:
        list[bool]: List of boolean values indicating non-null states.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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
    """
    Filter frames based on a specified quantity.

    Args:
        quantity: The quantity to filter on.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where filtered frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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
    """
    Shuffle the order of frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where shuffled frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
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
    return unpack_i(future, 0), unpack_i(future, 1)


@typeguard.typechecked
def get_index_element_mask(
    numbers: np.ndarray,
    atom_indices: Optional[list[int]],
    elements: Optional[list[str]],
    natoms_padded: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a mask for atom indices and elements.

    Args:
        numbers: Array of atomic numbers.
        atom_indices: List of atom indices to include.
        elements: List of element symbols to include.
        natoms_padded: Total number of atoms including padding.

    Returns:
        np.ndarray: Boolean mask array.
    """
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
