import io
import pickle
import copy
from pathlib import Path
from typing import Optional, Union, Any, Final
from collections.abc import Sequence

import ase.calculators.calculator
import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols, atomic_numbers
from ase.io.extxyz import key_val_dict_to_str, read_xyz, key_val_str_to_dict

import psiflow

# TODO: docstrings
# TODO: custom attributes writing
# TODO: Reference and Sampling want to add attributes to Geometry (used to be 'order' dict)
#  what to do?

# these are always accessible for every geometry
DEFAULT_PROPERTIES = "per_atom", "cell", "energy", "stress"
PER_ATOM_FIELDS = "numbers", "positions", "forces"


class MissingType:
    """Placeholder sentinel for missing data fields - replaces None"""

    def __repr__(self):
        return "<MISSING>"

    def __bool__(self):
        return False


MISSING: Final = MissingType()


class PerAtom:
    """
    Holds 'per atom' arrays like positions and forces
    For every attribute, the first array dimension should be n_atoms
    """

    numbers: npt.NDArray[np.uint8]
    positions: npt.NDArray[np.float64]
    forces: npt.NDArray[np.float64]

    def __init__(
        self,
        numbers: npt.NDArray[np.uint8],
        positions: npt.NDArray[np.float64],
        **kwargs: npt.NDArray,
    ):
        self.numbers = numbers.astype(int).reshape(-1, 1)
        self.positions = positions
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if value is MISSING:
            return  # do not set MISSING values
        elif not self._check(value):
            raise ValueError(f"Field '{name}' is not a per-atom property..")
        super().__setattr__(name, value)

    def __getattr__(self, name):
        # only runs if the attribute isn't found normally
        if name == "forces":
            return MISSING  # forces should always be accessible
        raise AttributeError(
            f"'{type(self).__name__}' instance has no attribute '{name}'"
        )

    def __len__(self) -> int:
        return self.numbers.shape[0]

    def __str__(self) -> str:
        return f"PerAtom[{len(self)}]{set(vars(self))}"

    def _check(self, arr: npt.NDArray) -> bool:
        try:
            return (arr.shape[0] == len(self)) and arr.ndim == 2
        except AttributeError:
            return arr.ndim == 2  # numbers not yet defined

    def reset(self):
        """Wipes optional fields"""
        attrs = [k for k in vars(self) if k not in ("numbers", "positions")]
        for attr in attrs:
            delattr(self, attr)

    def to_string(self) -> tuple[str, str]:
        symbols = [chemical_symbols[i] for i in self.numbers.flatten()]
        data = {"species": np.array(symbols).reshape(-1, 1), "pos": self.positions}
        if not self.forces is MISSING:
            data["forces"] = self.forces

        # create extxyz header
        header = "Properties=species:S:1:pos:R:3"
        for k, arr in data.items():
            if k in header:
                continue
            header += f":{k}:R:{arr.shape[1]}"

        # create a structured array with all fields
        dtypes = []
        for k, arr in data.items():
            dtypes += [arr.dtype] * arr.shape[1]
        dtype = [(str(i), t) for i, t in enumerate(dtypes)]

        structured = np.zeros(len(self), dtype=dtype)
        i = 0
        for k, arr in data.items():
            for j in range(arr.shape[1]):
                structured[str(i)] = arr[:, j]
                i += 1

        # write to string
        formats = []
        format_map = {"species": "%-2s"}
        for k, arr in data.items():
            dformat = format_map.get(k, "%16.8f")
            formats += [dformat] * arr.shape[1]
        buffer = io.StringIO()
        np.savetxt(buffer, structured, fmt=formats)
        txt = buffer.getvalue()

        return header, txt

    @classmethod
    def from_string(cls, s: str, props: Optional[str] = None):
        props = (props or "species:S:1:pos:R:3") + ":"
        dtype_map = {"R": "f8", "S": "i8"}

        # figure out what columns belong to which property
        keys, dtypes = [], []
        while props:
            k, t, n, props = props.split(":", 3)
            keys += [k] * int(n)
            dtypes += [dtype_map[t]] * int(n)
        dtype = [(str(i), t) for i, t in enumerate(dtypes)]

        # convert species to numbers automatically
        conv = {0: lambda s: atomic_numbers[s]}
        structured = np.loadtxt(io.StringIO(s), dtype=dtype, converters=conv)

        # group subarrays per property
        data = {k: [] for k in set(keys)}
        for i, k in enumerate(structured.dtype.names):
            arr = structured[k]
            data[keys[i]].append(arr)
        arrays = {k: np.stack(v, axis=-1) for k, v in data.items()}

        arrays["numbers"] = arrays.pop("species")
        arrays["positions"] = arrays.pop("pos")
        return cls(**arrays)


class Geometry:
    """
    Represents an atomic structure with associated properties.
    This class encapsulates the atomic structure, including atom positions, cell parameters,
    and various physical properties such as energy and forces.

    Attributes:
        per_atom (PerAtom): Record array containing per-atom properties.
        cell (np.ndarray): 3x3 array representing the unit cell vectors.
        energy (Optional[float]): Total energy of the system.
        stress (Optional[np.ndarray]): Stress tensor of the system.
    """

    per_atom: PerAtom
    cell: Optional[np.ndarray]
    energy: float
    stress: np.ndarray

    def __init__(
        self,
        per_atom: PerAtom,
        cell: Optional[np.ndarray] = None,
        **kwargs: Any,
    ):
        """
        Initialize a Geometry instance, though the preferred way of instantiating
        proceeds via the class methods
        """
        self.per_atom = per_atom
        self.cell = cell

        # set optional attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if value is MISSING:
            return  # do not set MISSING values

        # enforce format for DEFAULT_PROPERTIES
        if name == "cell" and value is not None:
            assert isinstance(value, np.ndarray) and value.shape == (3, 3)
        elif name == "energy":
            value = float(value)
            # assert isinstance(value, float)
        elif name == "stress":
            assert isinstance(value, np.ndarray) and value.shape == (3, 3)

        super().__setattr__(name, value)

    def __getattr__(self, name):
        # only runs if the attribute isn't found normally
        if name in ("energy", "stress"):
            return MISSING  # should always be accessible
        raise AttributeError(
            f"'{type(self).__name__}' instance has no attribute '{name}'"
        )

    def attributes(self) -> dict[str, Any]:
        """Return all non-MISSING instance attributes"""

        return {k: v for k, v in vars(self).items() if k != "per_atom"}

    def reset(self) -> None:
        """
        Reset all computed properties of the geometry to their default values.
        """
        self.per_atom.reset()
        for k in {"energy", "stress"}.intersection(self.attributes()):
            delattr(self, k)

    def clean(self) -> None:
        """
        Clean the geometry by resetting properties and removing additional information.
        """
        self.per_atom.reset()
        for k in self.attributes():
            if k == "cell":
                continue
            delattr(self, k)

    def align_axes(self) -> None:
        """
        Align the axes of the unit cell to a canonical representation for periodic systems.
        """
        if not self.periodic:
            return
        positions = self.per_atom.positions
        cell = self.cell
        transform_lower_triangular(positions, cell, reorder=False)
        reduce_box_vectors(cell)

    def copy(self) -> Geometry:
        """
        Create a deep copy of the Geometry instance.
        """
        return pickle.loads(pickle.dumps(self))

    def to_string(self) -> str:
        """
        Convert the Geometry instance to a string representation in extended XYZ format.
        """
        data = self.attributes()
        data.pop("cell")  # cell needs special treatment
        if self.periodic:
            data["Lattice"] = self.cell.T  # ase does Fortran ordering
            data["pbc"] = "T T T"
        else:
            data["pbc"] = "F F F"
        info_str = key_val_dict_to_str(data)
        properties, txt = self.per_atom.to_string()
        header = " ".join([properties, info_str])
        return "\n".join([str(len(self)), header, txt])

    def save(self, path_xyz: Path | str) -> None:
        """
        Save the Geometry instance to an XYZ file.
        """
        path_xyz = psiflow.resolve_and_check(Path(path_xyz))
        path_xyz.write_text(self.to_string())

    def to_atoms(self, structural_only: bool = False) -> Atoms:
        """
        Convert the Geometry instance to an Atoms object.
        """
        if structural_only:
            return Atoms(
                positions=self.per_atom.positions,
                numbers=self.numbers,
                cell=self.cell,
                pbc=self.periodic,
            )

        # use the ASE extxyz reader
        s = self.to_string()
        return next(read_xyz(io.StringIO(s), index=0))

    def __eq__(self, other: "Geometry") -> bool:
        """
        Check if two Geometry instances are structurally equal.
        """
        if (
            isinstance(other, Geometry)
            and len(self) == len(other)
            and self.periodic == other.periodic
            and (not self.periodic or np.allclose(self.cell, other.cell))
            and np.allclose(self.per_atom.numbers, other.per_atom.numbers)
            and np.allclose(self.per_atom.positions, other.per_atom.positions)
        ):
            return True
        return False

    def __len__(self) -> int:
        """
        Get the number of atoms in the geometry.
        """
        return len(self.per_atom)

    @classmethod
    def from_string(cls, s: str) -> Geometry:
        """
        Create a Geometry instance from a string representation in extended XYZ format.
        """
        n_atoms, header, body = s.strip().split("\n", 2)
        data = key_val_str_to_dict(header)
        data.pop("pbc", None)  # geometry derives pbc from cell
        if "Lattice" in data:
            data["cell"] = data.pop("Lattice").T  # ase does Fortran ordering
        else:
            data["cell"] = None
        per_atom = PerAtom.from_string(body, data.pop("Properties", None))

        assert len(per_atom) == int(n_atoms)
        return Geometry(per_atom, **data)

    @classmethod
    def load(cls, path_xyz: Path | str) -> "Geometry":
        """
        Load a Geometry instance from an XYZ file.
        """
        path_xyz = psiflow.resolve_and_check(Path(path_xyz))
        content = path_xyz.read_text()
        return cls.from_string(content)

    @classmethod
    def from_data(
        cls, numbers: np.ndarray, positions: np.ndarray, cell: Optional[np.ndarray]
    ) -> Geometry:
        """
        Create a Geometry instance from atomic numbers, positions, and cell data.
        """
        per_atom = PerAtom(numbers.copy(), positions.copy())
        return Geometry(per_atom, cell=copy.copy(cell))

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> Geometry:
        """
        Create a Geometry instance from an ASE Atoms object.
        """
        per_atom = PerAtom(**atoms.arrays)
        data = atoms.info
        if all(atoms.pbc):
            data["cell"] = atoms.cell.array
        if atoms.calc is not None:
            # ASE does stupid calc things
            try:
                data["energy"] = atoms.get_potential_energy()
                per_atom.forces = atoms.get_forces()
                data["stress"] = atoms.get_stress(voigt=False)
            except ase.calculators.calculator.PropertyNotImplementedError:
                pass  # property was not stored in calc
        return cls(per_atom, **data)

    @property
    def periodic(self) -> bool:
        """
        Check if the geometry is periodic.
        """
        return not self.cell is None

    @property
    def per_atom_energy(self) -> float | MISSING:
        """
        Calculate the energy per atom.
        """
        if self.energy is MISSING:
            return MISSING
        return self.energy / len(self)

    @property
    def volume(self) -> Optional[float]:
        """
        Calculate the volume of the unit cell.
        """
        if not self.periodic:
            return None
        return np.linalg.det(self.cell)

    @property
    def atomic_masses(self) -> npt.NDArray:
        """
        Get the atomic masses of the atoms in the geometry.
        """
        return np.array([atomic_masses[n] for n in self.numbers])

    @property
    def numbers(self) -> npt.NDArray:
        """
        Get a flattened version of the per_atom.numbers array.
        """
        return self.per_atom.numbers.flatten()


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


def get_atomic_energy(geometry: Geometry, atomic_energies: dict[str, float]) -> float:
    """Compute the total atomic energy based on provided single atom energies."""
    total = 0
    numbers, counts = np.unique(geometry.numbers, return_counts=True)
    for number, count in zip(numbers, counts):
        symbol = chemical_symbols[number]
        try:
            total += count * atomic_energies[symbol]
        except KeyError:
            raise KeyError(f"No atomic energy value for symbol '{symbol}'..")
    return float(total)
