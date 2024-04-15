from __future__ import annotations  # necessary for type-guarding class methods

from pathlib import Path
from typing import IO, Optional, Union

import numpy as np
import typeguard
from ase import Atoms
from ase.data import chemical_symbols
from ase.io.extxyz import key_val_dict_to_str, key_val_str_to_dict_regex
from parsl.app.app import python_app

from psiflow.data.utils import reduce_box_vectors, transform_lower_triangular

per_atom_dtype = np.dtype(
    [
        ("numbers", np.uint8),
        ("positions", np.float32, (3,)),
        ("forces", np.float32, (3,)),
    ]
)


@typeguard.typechecked
class Geometry:
    per_atom: np.recarray
    cell: np.ndarray
    order: dict
    energy: Optional[float]
    stress: Optional[np.ndarray]
    delta: Optional[float]
    phase: Optional[str]
    logprob: Optional[np.ndarray]
    stdout: Optional[str]
    identifier: Optional[int]

    def __init__(
        self,
        per_atom: np.recarray,
        cell: np.ndarray,
        order: Optional[dict] = None,
        energy: Optional[float] = None,
        stress: Optional[np.ndarray] = None,
        delta: Optional[float] = None,
        phase: Optional[str] = None,
        logprob: Optional[np.ndarray] = None,
        stdout: Optional[str] = None,
        identifier: Optional[int] = None,
    ):
        self.per_atom = per_atom.astype(per_atom_dtype)  # copies data
        self.cell = cell.astype(np.float32)
        assert self.cell.shape == (3, 3)
        if order is None:
            order = {}
        self.order = order
        self.energy = energy
        self.stress = stress
        self.delta = delta
        self.phase = phase
        self.logprob = logprob
        self.stdout = stdout
        self.identifier = identifier

    def reset(self):
        self.energy = None
        self.stress = None
        self.delta = None
        self.phase = None
        self.logprob = None
        self.per_atom.forces[:] = np.nan

    def clean(self):
        self.reset()
        self.order = {}
        self.stdout = None
        self.identifier = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, Geometry):
            return False
        # have to check separately for np.allclose due to different dtypes
        equal = True
        equal = equal and (len(self) == len(other))
        equal = equal and (self.periodic == other.periodic)
        if not equal:
            return False
        equal = equal and np.allclose(self.per_atom.numbers, other.per_atom.numbers)
        equal = equal and np.allclose(self.per_atom.positions, other.per_atom.positions)
        equal = equal and np.allclose(self.cell, other.cell)
        return bool(equal)

    def align_axes(self):
        if self.periodic:  # only do something if periodic:
            positions = self.per_atom.positions
            cell = self.cell
            transform_lower_triangular(positions, cell, reorder=False)
            reduce_box_vectors(cell)

    @classmethod
    def read(path: Union[str, Path], index: int = -1):
        path = Path(path)

    @property
    def periodic(self):
        return np.any(self.cell)

    @classmethod
    def from_data(
        cls,
        numbers: np.ndarray,
        positions: np.ndarray,
        cell: Optional[np.ndarray],
    ) -> Geometry:
        per_atom = np.recarray(len(numbers), dtype=per_atom_dtype)
        per_atom.numbers[:] = numbers
        per_atom.positions[:] = positions
        per_atom.forces[:] = np.nan
        if cell is not None:
            cell = cell.copy()
        else:
            cell = np.zeros((3, 3))
        return Geometry(per_atom, cell)

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> Geometry:
        per_atom = np.recarray(len(atoms), dtype=per_atom_dtype)
        per_atom.numbers[:] = atoms.numbers.astype(np.uint8)
        per_atom.positions[:] = atoms.get_positions()
        per_atom.forces[:] = atoms.arrays.get("forces", np.nan)
        if np.any(atoms.pbc):
            cell = np.array(atoms.cell)
        else:
            cell = np.zeros((3, 3))
        geometry = cls(per_atom, cell)
        geometry.energy = atoms.info.get("energy", None)
        geometry.stress = atoms.info.get("stress", None)
        geometry.delta = atoms.info.get("delta", None)
        geometry.phase = atoms.info.get("phase", None)
        geometry.logprob = atoms.info.get("logprob", None)
        geometry.stdout = atoms.info.get("stdout", None)
        geometry.identifier = atoms.info.get("identifier", None)
        return geometry

    def __len__(self):
        return len(self.per_atom)


def new_nullstate():
    return Geometry.from_data(np.zeros(1), np.zeros((1, 3)), None)


# use universal dummy state
NullState = new_nullstate()


@typeguard.typechecked
def _write_frames(*states: Geometry, outputs: list = []) -> None:
    with open(outputs[0], "w") as f:
        for i, state in enumerate(states):
            if state.periodic:
                comment = 'Lattice="'
                comment += " ".join(
                    [str(x) for x in np.reshape(state.cell.T, 9, order="F")]
                )
                comment += '" pbc="T T T" '
            else:
                comment = ""

            write_forces = not np.any(np.isnan(state.per_atom.forces))
            comment += "Properties=species:S:1:pos:R:3"
            if write_forces:
                comment += ":forces:R:3"
            comment += " "

            keys = [
                "energy",
                "stress",
                "delta",
                "phase",
                "logprob",
                "stdout",
                "identifier",
            ]
            values_dict = {}
            for key in keys:
                value = getattr(state, key)
                if value is not None:
                    values_dict[key] = value
            for key, value in state.order.items():
                values_dict["order_" + key] = value
            comment += key_val_dict_to_str(values_dict)
            f.write("{}\n".format(len(state)))
            f.write("{}\n".format(comment))
            fmt = " ".join(["%2s"] + 3 * ["%16.8f"]) + " "
            if write_forces:
                fmt += " ".join(3 * ["%16.8f"])
            fmt += "\n"
            for i in range(len(state)):
                entry = (chemical_symbols[state.per_atom.numbers[i]],)
                entry = entry + tuple(state.per_atom.positions[i])
                if write_forces:
                    entry = entry + tuple(state.per_atom.forces[i])
                f.write(fmt % entry)


write_frames = python_app(_write_frames, executors=["default_threads"])


@typeguard.typechecked
def _read_frame(f: IO, natoms: int) -> Geometry:
    comment = f.readline()
    comment_dict = key_val_str_to_dict_regex(comment)

    # read and format per_atom data
    per_atom = np.recarray(natoms, dtype=per_atom_dtype)
    for i in range(natoms):
        values = f.readline().split()
        per_atom.numbers[i] = chemical_symbols.index(values[0])
        per_atom.positions[i, :] = [float(_) for _ in values[1:4]]
        if len(values) > 4:
            per_atom.forces[i, :] = [float(_) for _ in values[4:7]]

    order = {}
    for key, value in comment_dict.items():
        if key.startswith("order_"):
            order[key.replace("order_", "")] = value

    geometry = Geometry(
        per_atom=per_atom,
        cell=comment_dict.pop("Lattice", np.zeros((3, 3))).T,  # transposed!
        energy=comment_dict.pop("energy", None),
        stress=comment_dict.pop("stress", None),
        delta=comment_dict.pop("delta", None),
        phase=comment_dict.pop("phase", None),
        logprob=comment_dict.pop("logprob", None),
        stdout=comment_dict.pop("stdout", None),
        identifier=comment_dict.pop("identifier", None),
        order=order,
    )
    return geometry


@typeguard.typechecked
def _read_frames(
    indices: Union[None, slice, list[int], int] = None,
    # safe: bool = False,
    inputs: list = [],
    outputs: list = [],
) -> Optional[list[Geometry]]:
    from psiflow.data.geometry import _write_frames
    from psiflow.data.utils import _count_frames

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
                data.append(_read_frame(f, natoms))
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
    from psiflow.data.geometry import _read_frames

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
    nframes = len(data)
    nprob = 0
    for state in data:
        if state.logprob is not None:
            nprob = max(len(state.logprob), nprob)
    arrays = []
    for quantity in quantities:
        if quantity in ["positions", "forces"]:
            array = np.empty((nframes, np.max(natoms), 3), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["cell", "stress"]:
            array = np.empty((nframes, 3, 3), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["numbers"]:
            array = np.empty((nframes, np.max(natoms), 1), dtype=np.uint8)
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
        for j, quantity in enumerate(quantities):
            if quantity == "positions":
                arrays[j][i, :, :] = geometry.per_atom.positions
            elif quantity == "forces":
                n = len(geometry)
                arrays[j][i, :n, :] = geometry.per_atom.forces
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
    from psiflow.data import NullState

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
    from psiflow.data.geometry import _assign_identifier, _read_frames, _write_frames

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
