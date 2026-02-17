import os
import time
from typing import Any, Optional

import numpy as np
from ase.data import chemical_symbols

# do not use psiflow apps; parsl config is not loaded in this process!
from psiflow.geometry import Geometry


class ForceMagnitudeException(Exception):
    pass


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def create_xml_list(items: list[str]) -> str:
    """Pure helper"""
    return " [ {} ] ".format(", ".join(items))


def check_forces(
    forces: np.ndarray,
    geometry: Any,
    max_force: float,
):
    if not isinstance(geometry, Geometry):
        geometry = Geometry.from_atoms(geometry)

    exceeded = np.linalg.norm(forces, axis=1) > max_force
    if not np.sum(exceeded):
        return
    indices = np.arange(len(geometry))[exceeded]
    numbers = geometry.per_atom.numbers[exceeded]
    symbols = [chemical_symbols[n] for n in numbers]
    raise ForceMagnitudeException(
        "\nforce exceeded {} eV/A for atoms {}"
        " with chemical elements {}\n".format(
            max_force,
            indices,
            symbols,
        )
    )


def initialise_driver(driver: 'Psiflow_driver') -> None:
    """"""
    function = driver.function
    name = function.__class__.__name__
    affinity = os.sched_getaffinity(os.getpid())
    t0 = time.time()
    for _ in range(10):
        function(driver.geometry)  # torch warm-up before simulation
    t1 = time.time()
    msg = [
        "- Psiflow -",
        f"Initialising driver for {name} with options {driver.kwargs}",
        f"CPU affinity [PID {os.getpid()}]: {affinity}",
        f"Time for 10 evaluations: {t1 - t0:.3f}",
        "- - - - - -",
    ]
    print("\n".join(msg))


def check_output(driver: 'Psiflow_driver', data: dict) -> None:
    """"""
    if max_force := driver.kwargs.get("max_force"):
        check_forces(data["forces"], driver.geometry, max_force)
