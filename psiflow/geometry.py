from __future__ import annotations  # necessary for type-guarding class methods

from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.io.extxyz import key_val_dict_to_str, key_val_str_to_dict_regex
from parsl.app.app import python_app

import psiflow

per_atom_dtype = np.dtype(
    [
        ("numbers", np.uint8),
        ("positions", np.float32, (3,)),
        ("forces", np.float32, (3,)),
    ]
)

QUANTITIES = [
    "positions",
    "cell",
    "numbers",
    "energy",
    "per_atom_energy",
    "forces",
    "stress",
    "delta",
    "logprob",
    "phase",
    "identifier",
]


@typeguard.typechecked
class Geometry:
    """
    Represents an atomic structure with associated properties.

    This class encapsulates the atomic structure, including atom positions, cell parameters,
    and various physical properties such as energy and forces.

    Attributes:
        per_atom (np.recarray): Record array containing per-atom properties.
        cell (np.ndarray): 3x3 array representing the unit cell vectors.
        order (dict): Dictionary to store custom ordering information.
        energy (Optional[float]): Total energy of the system.
        stress (Optional[np.ndarray]): Stress tensor of the system.
        delta (Optional[float]): Delta value, if applicable.
        phase (Optional[str]): Phase information, if applicable.
        logprob (Optional[np.ndarray]): Log probability values, if applicable.
        stdout (Optional[str]): Standard output information, if applicable.
        identifier (Optional[int]): Unique identifier for the geometry.
    """

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
        """
        Initialize a Geometry instance, though the preferred way of instantiating
        proceeds via the `from_data` or `from_atoms` class methods

        Args:
            per_atom (np.recarray): Record array containing per-atom properties.
            cell (np.ndarray): 3x3 array representing the unit cell vectors.
            order (Optional[dict], optional): Custom ordering information. Defaults to None.
            energy (Optional[float], optional): Total energy of the system. Defaults to None.
            stress (Optional[np.ndarray], optional): Stress tensor of the system. Defaults to None.
            delta (Optional[float], optional): Delta value. Defaults to None.
            phase (Optional[str], optional): Phase information. Defaults to None.
            logprob (Optional[np.ndarray], optional): Log probability values. Defaults to None.
            stdout (Optional[str], optional): Standard output information. Defaults to None.
            identifier (Optional[int], optional): Unique identifier for the geometry. Defaults to None.
        """
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
        """
        Reset all computed properties of the geometry to their default values.
        """
        self.energy = None
        self.stress = None
        self.delta = None
        self.phase = None
        self.logprob = None
        self.per_atom.forces[:] = np.nan

    def clean(self):
        """
        Clean the geometry by resetting properties and removing additional information.
        """
        self.reset()
        self.order = {}
        self.stdout = None
        self.identifier = None

    def __eq__(self, other) -> bool:
        """
        Check if two Geometry instances are equal.

        Args:
            other: The other object to compare with.

        Returns:
            bool: True if the geometries are equal, False otherwise.
        """
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
        """
        Align the axes of the unit cell to a canonical representation for periodic systems.
        """
        if self.periodic:  # only do something if periodic:
            positions = self.per_atom.positions
            cell = self.cell
            transform_lower_triangular(positions, cell, reorder=False)
            reduce_box_vectors(cell)

    def __len__(self):
        """
        Get the number of atoms in the geometry.

        Returns:
            int: The number of atoms.
        """
        return len(self.per_atom)

    def to_string(self) -> str:
        """
        Convert the Geometry instance to a string representation in extended XYZ format.

        Returns:
            str: String representation of the geometry.
        """
        if self.periodic:
            comment = 'Lattice="'
            comment += " ".join([str(x) for x in np.reshape(self.cell.T, 9, order="F")])
            comment += '" pbc="T T T" '
        else:
            comment = 'pbc="F F F" '

        write_forces = not np.any(np.isnan(self.per_atom.forces))
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
            value = getattr(self, key)
            if value is None:
                continue
            if type(value) is np.ndarray:
                if np.all(np.isnan(value)):
                    continue
            values_dict[key] = value
        for key, value in self.order.items():
            values_dict["order_" + key] = value
        comment += key_val_dict_to_str(values_dict)
        lines = ["{}".format(len(self))]
        lines.append("{}".format(comment))
        fmt = " ".join(["%2s"] + 3 * ["%16.8f"]) + " "
        if write_forces:
            fmt += " ".join(3 * ["%16.8f"])
        for i in range(len(self)):
            entry = (chemical_symbols[self.per_atom.numbers[i]],)
            entry = entry + tuple(self.per_atom.positions[i])
            if write_forces:
                entry = entry + tuple(self.per_atom.forces[i])
            lines.append(fmt % entry)
        return "\n".join(lines)

    def save(self, path_xyz: Union[Path, str]):
        """
        Save the Geometry instance to an XYZ file.

        Args:
            path_xyz (Union[Path, str]): Path to save the XYZ file.
        """
        path_xyz = psiflow.resolve_and_check(path_xyz)
        with open(path_xyz, "w") as f:
            f.write(self.to_string())

    def copy(self) -> Geometry:
        """
        Create a deep copy of the Geometry instance.

        Returns:
            Geometry: A new Geometry instance with the same data.
        """
        return Geometry.from_string(self.to_string())

    @classmethod
    def from_string(cls, s: str, natoms: Optional[int] = None) -> Optional[Geometry]:
        """
        Create a Geometry instance from a string representation in extended XYZ format.

        Args:
            s (str): String representation of the geometry.
            natoms (Optional[int], optional): Number of atoms (if known). Defaults to None.

        Returns:
            Optional[Geometry]: A new Geometry instance, or None if the string is empty.
        """
        if len(s) == 0:
            return None
        if not natoms:  # natoms in s
            lines = s.strip().split("\n")
            natoms = int(lines[0])
            lines = lines[1:]
        else:
            lines = s.rstrip().split(
                "\n"
            )  # i-PI nonperiodic starts with empty -> rstrip!
        assert len(lines) == natoms + 1
        comment = lines[0]
        comment_dict = key_val_str_to_dict_regex(comment)

        # read and format per_atom data
        per_atom = np.recarray(natoms, dtype=per_atom_dtype)
        per_atom.forces[:] = np.nan
        for i in range(natoms):
            values = lines[i + 1].split()
            per_atom.numbers[i] = chemical_symbols.index(values[0])
            per_atom.positions[i, :] = [float(_) for _ in values[1:4]]
            if len(values) > 4:
                per_atom.forces[i, :] = [float(_) for _ in values[4:7]]

        order = {}
        for key, value in comment_dict.items():
            if key.startswith("order_"):
                order[key.replace("order_", "")] = value

        geometry = cls(
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

    @classmethod
    def load(cls, path_xyz: Union[Path, str]) -> Geometry:
        """
        Load a Geometry instance from an XYZ file.

        Args:
            path_xyz (Union[Path, str]): Path to the XYZ file.

        Returns:
            Geometry: A new Geometry instance loaded from the file.
        """
        path_xyz = psiflow.resolve_and_check(Path(path_xyz))
        assert path_xyz.exists()
        with open(path_xyz, "r") as f:
            content = f.read()
        return cls.from_string(content)

    @property
    def periodic(self):
        """
        Check if the geometry is periodic.

        Returns:
            bool: True if the geometry is periodic, False otherwise.
        """
        return np.any(self.cell)

    @property
    def per_atom_energy(self):
        """
        Calculate the energy per atom.

        Returns:
            Optional[float]: Energy per atom if total energy is available, None otherwise.
        """
        if self.energy is None:
            return None
        else:
            return self.energy / len(self)

    @property
    def volume(self):
        """
        Calculate the volume of the unit cell.

        Returns:
            float: Volume of the unit cell for periodic systems, np.nan for non-periodic systems.
        """
        if not self.periodic:
            return np.nan
        else:
            return np.linalg.det(self.cell)

    @classmethod
    def from_data(
        cls,
        numbers: np.ndarray,
        positions: np.ndarray,
        cell: Optional[np.ndarray],
    ) -> Geometry:
        """
        Create a Geometry instance from atomic numbers, positions, and cell data.

        Args:
            numbers (np.ndarray): Array of atomic numbers.
            positions (np.ndarray): Array of atomic positions.
            cell (Optional[np.ndarray]): Unit cell vectors (or None for non-periodic systems).

        Returns:
            Geometry: A new Geometry instance.
        """
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
        """
        Create a Geometry instance from an ASE Atoms object.

        Args:
            atoms (Atoms): ASE Atoms object.

        Returns:
            Geometry: A new Geometry instance.
        """
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


def new_nullstate():
    """
    Create a new null state Geometry.

    Returns:
        Geometry: A Geometry instance representing a null state.
    """
    return Geometry.from_data(np.zeros(1), np.zeros((1, 3)), None)


# use universal dummy state
NullState = new_nullstate()


def is_lower_triangular(cell: np.ndarray) -> bool:
    """
    Check if a cell matrix is lower triangular.

    Args:
        cell (np.ndarray): 3x3 cell matrix.

    Returns:
        bool: True if the cell matrix is lower triangular, False otherwise.
    """
    return (
        cell[0, 0] > 0
        and cell[1, 1] > 0  # positive volumes
        and cell[2, 2] > 0
        and cell[0, 1] == 0
        and cell[0, 2] == 0  # lower triangular
        and cell[1, 2] == 0
    )


def is_reduced(cell: np.ndarray) -> bool:
    """
    Check if a cell matrix is in reduced form.

    Args:
        cell (np.ndarray): 3x3 cell matrix.

    Returns:
        bool: True if the cell matrix is in reduced form, False otherwise.
    """
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

    Args:
        pos (np.ndarray): Array of atomic positions.
        cell (np.ndarray): 3x3 cell matrix.
        reorder (bool, optional): Whether to reorder lattice vectors. Defaults to False.
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
def get_mass_matrix(geometry: Geometry) -> np.ndarray:
    """
    Compute the mass matrix for a given geometry.

    Args:
        geometry (Geometry): Input geometry.

    Returns:
        np.ndarray: Mass matrix.
    """
    masses = np.repeat(
        np.array([atomic_masses[n] for n in geometry.per_atom.numbers]),
        3,
    )
    sqrt_inv = 1 / np.sqrt(masses)
    return np.outer(sqrt_inv, sqrt_inv)


@typeguard.typechecked
def mass_weight(hessian: np.ndarray, geometry: Geometry) -> np.ndarray:
    """
    Apply mass-weighting to a Hessian matrix.

    Args:
        hessian (np.ndarray): Input Hessian matrix.
        geometry (Geometry): Geometry associated with the Hessian.

    Returns:
        np.ndarray: Mass-weighted Hessian matrix.
    """
    assert hessian.shape[0] == hessian.shape[1]
    assert len(geometry) * 3 == hessian.shape[0]
    return hessian * get_mass_matrix(geometry)


@typeguard.typechecked
def mass_unweight(hessian: np.ndarray, geometry: Geometry) -> np.ndarray:
    """
    Remove mass-weighting from a Hessian matrix.

    Args:
        hessian (np.ndarray): Input mass-weighted Hessian matrix.
        geometry (Geometry): Geometry associated with the Hessian.

    Returns:
        np.ndarray: Unweighted Hessian matrix.
    """
    assert hessian.shape[0] == hessian.shape[1]
    assert len(geometry) * 3 == hessian.shape[0]
    return hessian / get_mass_matrix(geometry)


def create_outputs(quantities: list[str], data: list[Geometry]) -> list[np.ndarray]:
    """
    Create output arrays for specified quantities from a list of Geometry instances.

    Args:
        quantities (list[str]): List of quantity names to extract.
        data (list[Geometry]): List of Geometry instances.

    Returns:
        list[np.ndarray]: List of arrays containing the requested quantities.
    """
    order_names = list(set([k for g in data for k in g.order]))
    assert all([q in QUANTITIES + order_names for q in quantities])
    natoms = np.array([len(geometry) for geometry in data], dtype=int)
    max_natoms = np.max(natoms)
    nframes = len(data)
    nprob = 0
    max_phase = 0
    for state in data:
        if state.logprob is not None:
            nprob = max(len(state.logprob), nprob)
        if state.phase is not None:
            max_phase = max(len(state.phase), max_phase)

    arrays = []
    for quantity in quantities:
        if quantity in ["positions", "forces"]:
            array = np.empty((nframes, max_natoms, 3), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["cell", "stress"]:
            array = np.empty((nframes, 3, 3), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["numbers"]:
            array = np.empty((nframes, max_natoms), dtype=np.uint8)
            array[:] = 0
        elif quantity in ["energy", "delta", "per_atom_energy"]:
            array = np.empty((nframes,), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["phase"]:
            array = np.empty((nframes,), dtype=(np.unicode_, max_phase))
            array[:] = ""
        elif quantity in ["logprob"]:
            array = np.empty((nframes, nprob), dtype=np.float32)
            array[:] = np.nan
        elif quantity in ["identifier"]:
            array = np.empty((nframes,), dtype=np.int32)
            array[:] = -1
        elif quantity in order_names:
            array = np.empty((nframes,), dtype=np.float32)
            array[:] = np.nan
        else:
            raise AssertionError("missing quantity in if/else")
        arrays.append(array)
    return arrays


def _assign_identifier(
    state: Geometry,
    identifier: int,
    discard: bool = False,
) -> tuple[Geometry, int]:
    """
    Assign an identifier to a Geometry instance.

    Args:
        state (Geometry): Input Geometry instance.
        identifier (int): Identifier to assign.
        discard (bool, optional): Whether to discard the state. Defaults to False.

    Returns:
        tuple[Geometry, int]: Updated Geometry and next available identifier.
    """
    if (state == NullState) or discard:
        return state, identifier
    else:
        assert state.identifier is None
        state.identifier = identifier
        return state, identifier + 1


assign_identifier = python_app(_assign_identifier, executors=["default_threads"])


@typeguard.typechecked
def _check_equality(
    state0: Geometry,
    state1: Geometry,
) -> bool:
    """
    Check if two Geometry instances are equal.

    Args:
        state0 (Geometry): First Geometry instance.
        state1 (Geometry): Second Geometry instance.

    Returns:
        bool: True if the Geometry instances are equal, False otherwise.
    """
    return state0 == state1


check_equality = python_app(_check_equality, executors=["default_threads"])
