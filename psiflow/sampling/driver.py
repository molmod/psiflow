"""
Updated version of the Psiflow driver included in i-Pi
"""

import os
import time

import numpy as np
from ase.data import chemical_symbols
from ipi.pes.dummy import Dummy_driver
from ipi.utils.units import unit_to_internal, unit_to_user
from ipi.utils.messages import warning

try:
    from ase.io import read
    from psiflow.geometry import Geometry
    from psiflow.functions import function_from_json, Function
except ImportError as e:
    message = (
        "Could not find the Psiflow driver dependencies, "
        "make sure they are installed with "
        "`pip install git+https://github.com/molmod/psiflow.git`."
    )
    warning(f"{message}: {e}")
    raise ImportError(message) from e


__DRIVER_NAME__ = "psiflow"
__DRIVER_CLASS__ = "Psiflow_driver"


class Psiflow_driver(Dummy_driver):
    """
    Driver for Psiflow functions.
    The driver requires a template file (in ASE readable format) and a JSON file
    containing the function definition. Requires Psiflow to be installed.

    Command-line:
        i-pi-driver -m psiflow -o template=template.xyz,hamiltonian=hamiltonian.json

    Parameters:
        :param template: str, filename of an ASE-readable structure file
        :param hamiltonian: str, filename of a JSON file defining the Psiflow function
    """

    template: str
    hamiltonian: str
    geometry: Geometry
    function: Function

    def __init__(self, template, hamiltonian, *args, **kwargs):
        self.template = template
        self.hamiltonian = hamiltonian
        super().__init__(*args, **kwargs)

    def check_parameters(self):
        self.geometry = Geometry.from_atoms(read(self.template))
        self.function = function_from_json(self.hamiltonian, **self.kwargs)
        self.initialise()

    def compute_structure(self, cell, pos):
        pos = unit_to_user("length", "angstrom", pos)
        cell = unit_to_user("length", "angstrom", cell.T)
        self.geometry.per_atom.positions[:] = pos
        if self.geometry.periodic:
            self.geometry.cell[:] = cell

        outputs = self.function(self.geometry)
        self.check_output(outputs)

        # converts to internal quantities
        energy = outputs["energy"]
        forces = outputs["forces"]
        stress = outputs["stress"]
        pot_ipi = np.asarray(
            unit_to_internal("energy", "electronvolt", energy), np.float64
        )
        force_ipi = np.asarray(unit_to_internal("force", "ev/ang", forces), np.float64)
        if self.geometry.periodic:
            vir_calc = -stress * self.geometry.volume
        else:
            vir_calc = np.zeros_like(stress)
        vir_ipi = np.array(
            unit_to_internal("energy", "electronvolt", vir_calc.T), dtype=np.float64
        )
        extras = ""

        return pot_ipi, force_ipi, vir_ipi, extras

    def initialise(self):
        """"""
        function = self.function
        name = function.__class__.__name__
        affinity = os.sched_getaffinity(os.getpid())
        t0 = time.time()
        for _ in range(10):
            function(self.geometry)  # torch warm-up before simulation
        t1 = time.time()
        msg = [
            "- Psiflow -",
            f"Initialising driver for {name} with options {self.kwargs}",
            f"CPU affinity [PID {os.getpid()}]: {affinity}",
            f"Time for 10 evaluations: {t1 - t0:.3f}",
            "- - - - - -",
        ]
        print("\n".join(msg))

    def check_output(self, data: dict) -> None:
        if max_force := self.kwargs.get("max_force"):
            check_forces(data["forces"], self.geometry, max_force)


class ForceMagnitudeException(Exception):
    pass


def check_forces(forces: np.ndarray, geometry: Geometry, max_force: float):
    exceeded = np.linalg.norm(forces, axis=1) > max_force
    if not np.sum(exceeded):
        return
    indices = np.arange(len(geometry))[exceeded]
    numbers = geometry.numbers[exceeded]
    symbols = [chemical_symbols[n] for n in numbers]
    raise ForceMagnitudeException(
        "\nforce exceeded {} eV/A for atoms {}"
        " with chemical elements {}\n".format(max_force, indices, symbols)
    )
