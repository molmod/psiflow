"""
Updated version of the Psiflow driver included in i-Pi
"""

import numpy as np

from ipi.pes.dummy import Dummy_driver
from ipi.utils.units import unit_to_internal, unit_to_user
from ipi.utils.messages import warning

try:
    from ase.io import read
    from psiflow.geometry import Geometry
    from psiflow.functions import function_from_json, Function
    from psiflow.sampling.utils import initialise_driver, check_output
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
        initialise_driver(self)

    def compute_structure(self, cell, pos):
        pos = unit_to_user("length", "angstrom", pos)
        cell = unit_to_user("length", "angstrom", cell.T)
        self.geometry.per_atom.positions[:] = pos
        if self.geometry.periodic:
            self.geometry.cell[:] = cell

        outputs = self.function(self.geometry)
        check_output(self, outputs)

        # converts to internal quantities
        energy = outputs["energy"]
        forces = outputs["forces"]
        stress = outputs["stress"]
        pot_ipi = np.asarray(
            unit_to_internal("energy", "electronvolt", energy), np.float64
        )
        force_ipi = np.asarray(unit_to_internal("force", "ev/ang", forces), np.float64)
        vir_calc = -stress * self.geometry.volume
        vir_ipi = np.array(
            unit_to_internal("energy", "electronvolt", vir_calc.T), dtype=np.float64
        )
        extras = ""

        return pot_ipi, force_ipi, vir_ipi, extras
