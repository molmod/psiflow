from typing import Optional
from collections.abc import Sequence

import numpy as np
from ase.data import atomic_numbers

from psiflow.geometry import (
    Geometry,
    get_atomic_energy,
    DEFAULT_PROPERTIES,
    PER_ATOM_FIELDS,
    MISSING,
)


# TODO: some of these have side effects..


def extract(states: Sequence[Geometry], quantities: Sequence[str]) -> dict[str, list]:
    """
    Extract specified quantities from Geometry instances.
    """
    data = {}

    # figure out where to find quantities
    per_atom, per_structure, unknown = [], [], []
    for q in set(quantities):
        if q in PER_ATOM_FIELDS:
            per_atom.append(q)
        elif q in DEFAULT_PROPERTIES:
            per_structure.append(q)
        else:
            unknown.append(q)

    for q in per_atom:
        data[q] = [getattr(geom.per_atom, q, MISSING) for geom in states]
    for q in per_structure:
        data[q] = [getattr(geom, q, MISSING) for geom in states]
    for q in unknown:
        # try both options
        data[q] = []
        for geom in states:
            value = getattr(geom, q, MISSING)
            if value is MISSING:
                value = getattr(geom.per_atom, q, MISSING)
            data[q].append(value)

    return data


def insert(states: Sequence[Geometry], data: dict[str, list]) -> Sequence[Geometry]:
    """
    Insert quantities from data into Geometry instances.
    """
    for q, values in data.items():
        assert len(states) == len(values)
        for geom, v in zip(states, values):
            if isinstance(v, np.ndarray) and len(geom) == len(v):
                setattr(geom.per_atom, q, v)
            else:
                setattr(geom, q, v)
    return states


def extract_per_atom(
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


def filter_quantity(
    states: Sequence[Geometry],
    quantity: str,
) -> list[Geometry]:
    """
    Filter frames based on a specified quantity.
    """
    data = extract(states, [quantity])[quantity]
    return [geom for i, geom in enumerate(states) if not data[i] is MISSING]


def get_unique_numbers(states: Sequence[Geometry]) -> set[int]:
    """Returns a set of unique atom numbers found across all states."""
    numbers = set(el for geom in states for el in np.unique(geom.per_atom.numbers))
    return {int(n) for n in numbers}


def apply_energy_offset(
    states: Sequence[Geometry], subtract: bool, **atomic_energies: float
) -> None:
    """Apply an energy offset to all geometries"""
    numbers_data = get_unique_numbers(states)
    numbers_kwargs = {atomic_numbers[e] for e in atomic_energies.keys()}
    assert numbers_data == numbers_kwargs, "Provide atomic energies for all elements.."

    for geom in states:
        energy = get_atomic_energy(geom, atomic_energies)
        if subtract:
            geom.energy -= energy
        else:
            geom.energy += energy


def assign_ids(
    states: Sequence[Geometry], identifier: Optional[int] = None
) -> tuple[Sequence[Geometry], int]:
    """
    Assign unique identifiers to Geometry instances, starting at provided identifier.
    """
    # TODO: when would we want to supply the starting identifier?
    if identifier is None:
        # find largest existing value and add one
        ids = extract(states, ["identifier"])["identifier"]
        identifier = max([i for i in ids if i is not MISSING], default=0) + 1

    for geom in states:
        if hasattr(geom, "identifier"):
            continue
        geom.identifier = identifier
        identifier += 1

    return states, identifier
