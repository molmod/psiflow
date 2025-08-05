"""
Not sure whether this is necessary, maybe absorb into reference.py?
"""

from enum import Enum
from typing import Sequence

import numpy as np

from psiflow.geometry import Geometry


class Status(Enum):
    SUCCESS = 0
    FAILED = 1
    INCONSISTENT = 2


def find_line(
    lines: list[str],
    line: str,
    idx_start: int = 0,
    max_lines: int = int(1e6),
    reverse: bool = False,
) -> int | None:
    """"""
    if not reverse:
        idx_slice = slice(idx_start, idx_start + max_lines)
    else:
        idx_start = idx_start or len(lines) - 1
        idx_slice = slice(idx_start, idx_start - max_lines, -1)
    for i, l in enumerate(lines[idx_slice]):
        if l.strip().startswith(line):
            if not reverse:
                return idx_start + i
            else:
                return idx_start - i


def lines_to_array(
    lines: list[str], start: int = 0, stop: int = int(1e6), dtype: np.dtype = float
) -> np.ndarray:
    """"""
    return np.array([line.split()[start:stop] for line in lines], dtype=dtype)


def get_spin_multiplicities(element: str) -> list[int]:
    """TODO: rethink this"""
    # max S = N * 1/2, max mult = 2 * S + 1
    from ase.symbols import atomic_numbers

    mults = []
    number = atomic_numbers[element]
    for mult in range(1, min(number + 2, 16)):
        if number % 2 == 0 and mult % 2 == 0:
            continue  # S always whole, mult never even
        mults.append(mult)
    return mults


def copy_data_to_geometry(geom: Geometry, data: dict) -> Geometry:
    """"""
    geom = geom.copy()
    geom.reset()
    metadata = {k: data[k] for k in ("status", "stdout", "stderr")}
    geom.order |= metadata
    print(metadata)  # TODO: nice for debugging

    if data["status"] != Status.SUCCESS:
        return geom
    geom.order['runtime'] = data.get('runtime')

    if not np.allclose(data["positions"], geom.per_atom.positions, atol=1e-6):
        # output does not match geometry
        geom.order["status"] = Status.INCONSISTENT
        return geom

    geom.energy = data["energy"]
    if "forces" in data:
        geom.per_atom.forces[:] = data["forces"]

    return geom
