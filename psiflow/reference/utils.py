"""
Not sure whether this is necessary, maybe absorb into reference.py?
"""

from enum import Enum

import numpy as np

from psiflow.geometry import Geometry

class Status(Enum):
    SUCCESS = 0
    FAILED = 1


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


def copy_data_to_geometry(geom: Geometry, data: dict | None = None) -> Geometry:
    """"""
    # TODO: revamp this method
    # TODO: reject when data cannot match geometry
    geom = geom.copy()
    geom.reset()
    if data is None:
        return geom
    geom.order['status'] = data['status']
    geom.order['runtime'] = data['runtime']
    geom.energy = data['energy']
    if 'forces' in data:
        geom.per_atom.forces[:] = data['forces']
    return geom
