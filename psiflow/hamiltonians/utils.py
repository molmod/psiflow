import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import typeguard
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import fs, kJ, mol, nm
from parsl.data_provider.files import File

from psiflow.geometry import Geometry, chemical_symbols


class ForceMagnitudeException(Exception):
    pass


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


class EinsteinCalculator(Calculator):
    """ASE Calculator for a simple Einstein crystal"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        centers: np.ndarray,
        force_constant: float,
        volume: float,
        max_force: Optional[float] = None,
        **kwargs,
    ) -> None:
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.centers = centers
        self.force_constant = force_constant
        self.volume = volume
        self.max_force = max_force

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
        if self.max_force is not None:
            check_forces(forces, atoms, self.max_force)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }
        if sum(atoms.pbc) and self.volume > 0.0:
            delta = np.linalg.det(atoms.cell) - self.volume
            self.results["stress"] = self.force_constant * np.eye(3) * delta
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])
        else:
            self.results["stress"] = np.zeros(6)  # required by ASEDriver


@typeguard.typechecked
class PlumedCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        plumed_input: str,
        external: Union[str, Path, File, None] = None,
        max_force: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.plumed_input = plumed_input
        if external is not None:
            if isinstance(external, File):
                external = external.filepath
            assert Path(external).exists()
        self.external = external
        self.max_force = max_force

        self.tmp = tempfile.NamedTemporaryFile(
            prefix="plumed_", mode="w+", delete=False
        )
        # plumed creates a back up if this file would already exist
        os.remove(self.tmp.name)

        from plumed import Plumed

        plumed = Plumed()
        ps = 1000 * fs
        plumed.cmd("setMDEnergyUnits", mol / kJ)
        plumed.cmd("setMDLengthUnits", 1 / nm)
        plumed.cmd("setMDTimeUnits", 1 / ps)
        plumed.cmd("setMDChargeUnits", 1.0)
        plumed.cmd("setMDMassUnits", 1.0)

        # plumed.cmd("setMDEngine", "ASE")
        plumed.cmd("setLogFile", self.tmp.name)
        # plumed.cmd("setTimestep", float(timestep))
        plumed.cmd("setRestart", True)
        self.plumed = plumed
        self.initialize_plumed = True  # need natoms to initialize

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        if self.initialize_plumed:
            self.plumed.cmd("setNatoms", len(atoms))
            self.plumed.cmd("init")
            plumed_input = self.plumed_input
            for line in plumed_input.split("\n"):
                self.plumed.cmd("readInputLine", line)
            self.initialize_plumed = False
            os.remove(self.tmp.name)  # remove whatever plumed has created

        if atoms.pbc.any():
            self.plumed.cmd("setBox", np.asarray(atoms.cell))

        # set positions
        energy = np.zeros((1,))
        forces = np.zeros((len(atoms), 3))
        virial = np.zeros((3, 3))
        self.plumed.cmd("setStep", 0)
        self.plumed.cmd("setPositions", atoms.get_positions())
        self.plumed.cmd("setMasses", atoms.get_masses())
        self.plumed.cmd("setForces", forces)
        self.plumed.cmd("setVirial", virial)
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalcNoUpdate")
        self.plumed.cmd("getBias", energy)
        if all(atoms.pbc):
            stress = full_3x3_to_voigt_6_stress(virial / atoms.get_volume())
        else:
            stress = np.zeros(6)

        if self.max_force is not None:
            check_forces(forces, atoms, self.max_force)
        self.results = {
            "energy": energy[0],  # unpack to float
            "forces": forces,
            "stress": stress,
            "free_energy": energy[0],
        }


@typeguard.typechecked
class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        positions: np.ndarray,
        hessian: np.ndarray,
        energy: float,
        max_force: Optional[float] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        assert hessian.shape[0] == 3 * positions.shape[0]
        self.positions = positions
        self.hessian = hessian
        self.energy = energy
        self.max_force = max_force

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)
        assert self.hessian.shape[0] == 3 * len(atoms)

        pos = atoms.positions.reshape(-1)

        delta = pos - self.positions.reshape(-1)
        grad = np.dot(self.hessian, delta)
        energy = self.energy + 0.5 * np.dot(delta, grad)

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": (-1.0) * grad.reshape(-1, 3),
            "stress": np.zeros(6),
        }
        if sum(atoms.pbc):
            self.results["stress"] = full_3x3_to_voigt_6_stress(np.zeros((3, 3)))
