import numpy as np

# try:
#    from ase.calculators.calculator import BaseCalculator
# except ImportError:  # 3.22.1 and below still use Calculator
#    from ase.calculators.calculator import Calculator as BaseCalculator
from ase.calculators.calculator import Calculator, all_changes


class EinsteinCalculator(Calculator):
    """ASE Calculator for a simple Einstein crystal"""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, centers: np.ndarray, force_constant: float, **kwargs) -> None:
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.centers = centers
        self.force_constant = force_constant

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        assert self.centers.shape[0] == len(atoms)
        forces = (-1.0) * self.force_constant * (atoms.get_positions() - self.centers)
        energy = (
            self.force_constant
            / 2
            * np.sum((atoms.get_positions() - self.centers) ** 2)
        )
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }
