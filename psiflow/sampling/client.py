import argparse
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import typeguard
from ase.data import chemical_symbols
from ase.io import read

from psiflow.functions import Function, function_from_json

# do not use psiflow apps; parsl config is not loaded in this process!
from psiflow.geometry import Geometry

# only import stuff which does not issue useless warnings; otherwise
# python -c 'import .client; print(client.__file__)' is going to be polluted
# with those import-related warnings


class ForceMagnitudeException(Exception):
    pass


@typeguard.typechecked
def check_forces(
    forces: np.ndarray,
    geometry: Any,
    max_force: float,
):
    if not isinstance(geometry, Geometry):
        geometry = Geometry.from_atoms(geometry)

    exceeded = np.linalg.norm(forces, axis=1) > max_force
    if np.sum(exceeded):
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
    else:
        pass


class FunctionDriver:

    def __init__(
        self,
        template: Geometry,
        function: Function,
        max_force: Optional[float],
        verbose: bool = True,  # used by i-PI internally?
        error_msg="",
    ):
        self.verbose = verbose
        self.template = template
        self.function = function
        self.max_force = max_force

    def check_arguments(self):
        pass

    def __call__(self, cell, pos):
        from ipi.utils.units import unit_to_internal, unit_to_user

        pos = unit_to_user("length", "angstrom", pos)
        cell = unit_to_user("length", "angstrom", cell.T)

        self.template.per_atom.positions[:] = pos
        if self.template.periodic:
            self.template.cell[:] = cell

        outputs = self.function([self.template])

        energy = outputs["energy"][0]
        forces = outputs["forces"][0]
        stress = outputs["stress"][0]

        # check for max_force
        if self.max_force is not None:
            check_forces(forces, self.template, self.max_force)

        # converts to internal quantities
        pot_ipi = np.asarray(
            unit_to_internal("energy", "electronvolt", energy), np.float64
        )
        force_ipi = np.asarray(unit_to_internal("force", "ev/ang", forces), np.float64)
        vir_calc = -stress * self.template.volume
        vir_ipi = np.array(
            unit_to_internal("energy", "electronvolt", vir_calc.T), dtype=np.float64
        )
        extras = ""

        return pot_ipi, force_ipi, vir_ipi, extras


if __name__ == "__main__":
    print("OS environment values:")
    for key, value in os.environ.items():
        print(key, value)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_hamiltonian",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_force",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    assert args.path_hamiltonian is not None
    assert args.address is not None
    assert args.start is not None

    template = Geometry.from_atoms(read(args.start))
    function = function_from_json(
        args.path_hamiltonian,
        device=args.device,
        dtype=args.dtype,
    )

    t0 = time.time()
    function([template] * 10)  # torch warmp-up before simulation
    print('time for 10 evaluations: {}'.format(time.time() - t0))

    driver = FunctionDriver(
        template=template,
        function=function,
        max_force=args.max_force,
        verbose=True,
    )

    import psutil
    import torch
    from ipi._driver.driver import run_driver

    print("pid: {}".format(os.getpid()))
    print("CPU affinity: {}".format(os.sched_getaffinity(os.getpid())))
    print("torch num threads: ", torch.get_num_threads())
    print("cpu count: ", psutil.cpu_count(logical=False))

    try:
        run_driver(
            unix=True,
            address=str(Path.cwd() / args.address),
            driver=driver,
            sockets_prefix="",
        )
    except ForceMagnitudeException as e:
        print(e)  # induce timeout in server
    except ConnectionResetError as e:  # some other client induced a timeout
        print(e)
