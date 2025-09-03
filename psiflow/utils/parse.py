import datetime
from pathlib import Path

import numpy as np

from psiflow.execution import PSIFLOW_INTERNAL


class LineNotFoundError(Exception):
    """Call to find_line failed"""

    pass


def find_line(
    lines: list[str],
    line: str,
    idx_start: int = 0,
    max_lines: int = int(1e6),
    reverse: bool = False,
) -> int:
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
    raise LineNotFoundError('Could not find line starting with "{}".'.format(line))


def lines_to_array(
    lines: list[str], start: int = 0, stop: int = int(1e6), dtype: np.dtype = float
) -> np.ndarray:
    """"""
    return np.array([line.split()[start:stop] for line in lines], dtype=dtype)


def string_to_timedelta(timedelta: str) -> datetime.timedelta:
    """"""
    allowed_units = "weeks", "days", "hours", "minutes", "seconds"
    time_list = timedelta.split()
    values, units = time_list[:-1:2], time_list[1::2]
    kwargs = {u: float(v) for u, v in zip(units, values) if u in allowed_units}
    return datetime.timedelta(**kwargs)


def get_task_logs(task_id: int) -> tuple[Path, Path]:
    """"""
    path = Path.cwd().resolve() / PSIFLOW_INTERNAL / "000/task_logs"  # TODO
    stdout = next(path.rglob(f"task_{task_id}_*.stdout"))
    stderr = next(path.rglob(f"task_{task_id}_*.stderr"))
    return stdout, stderr
