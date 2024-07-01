import os
import tempfile
from pathlib import Path
from typing import Optional, ClassVar, Union
from dataclasses import dataclass

from ase.units import kJ, mol, nm, fs
from ase.data import atomic_masses
import typeguard
import numpy as np

from psiflow.geometry import Geometry, create_outputs, NullState


@typeguard.typechecked
@dataclass
class Function:
    outputs: ClassVar[tuple]

    def __call__(
        self,
        geometries: list[Geometry],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError


@dataclass
class EnergyFunction(Function):
    outputs: ClassVar[tuple[str, ...]] = ('energy', 'forces', 'stress')


@typeguard.typechecked
@dataclass
class EinsteinCrystalFunction(EnergyFunction):
    force_constant: float
    centers: np.ndarray
    volume: float = 0.0

    def __call__(
        self,
        geometries: list[Geometry],
    ) -> dict[str, np.ndarray]:
        value, grad_pos, grad_cell = create_outputs(
            self.outputs,
            geometries,
        )

        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue

            delta = geometry.per_atom.positions - self.centers
            value[i] = self.force_constant * np.sum(delta ** 2) / 2
            grad_pos[i, :len(geometry)] = (-1.0) * self.force_constant * delta
            if geometry.periodic and self.volume > 0.0:
                delta = np.linalg.det(geometry.cell) - self.volume
                _stress = self.force_constant * np.eye(3) * delta
            else:
                _stress = np.zeros((3, 3))
            grad_cell[i, :] = _stress

        return {'energy': value, 'forces': grad_pos, 'stress': grad_cell}


@typeguard.typechecked
@dataclass
class PlumedFunction(EnergyFunction):
    plumed_input: str
    external: Optional[Union[str, Path]] = None

    def __call__(
        self,
        geometries: list[Geometry],
    ) -> dict[str, np.ndarray]:
        value, grad_pos, grad_cell = create_outputs(
            self.outputs,
            geometries,
        )
        if self.external is None:
            plumed_input = self.plumed_input
        else:
            plumed_input = self.plumed_input.replace("PLACEHOLDER", str(self.external))

        def geometry_to_key(geometry: Geometry) -> tuple:
            return tuple([geometry.periodic]) + tuple(geometry.per_atom.numbers)

        plumed_instances = {}
        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue
            key = geometry_to_key(geometry)
            if key not in plumed_instances:
                from plumed import Plumed
                tmp = tempfile.NamedTemporaryFile(
                    prefix="plumed_", mode="w+", delete=False
                )
                # plumed creates a back up if this file would already exist
                os.remove(tmp.name)
                plumed_ = Plumed()
                ps = 1000 * fs
                plumed_.cmd("setRealPrecision", 8)
                plumed_.cmd("setMDEnergyUnits", mol / kJ)
                plumed_.cmd("setMDLengthUnits", 1 / nm)
                plumed_.cmd("setMDTimeUnits", 1 / ps)
                plumed_.cmd("setMDChargeUnits", 1.0)
                plumed_.cmd("setMDMassUnits", 1.0)

                plumed_.cmd("setLogFile", tmp.name)
                plumed_.cmd("setRestart", True)
                plumed_.cmd("setNatoms", len(geometry))
                plumed_.cmd("init")
                plumed_input = self.plumed_input
                for line in plumed_input.split("\n"):
                    plumed_.cmd("readInputLine", line)
                os.remove(tmp.name)  # remove whatever plumed has created
                plumed_instances[key] = plumed_

            plumed_ = plumed_instances[key]
            if geometry.periodic:
                cell = np.copy(geometry.cell).astype(np.float64)
                plumed_.cmd("setBox", cell)

            # set positions
            energy = np.zeros((1,))
            forces = np.zeros((len(geometry), 3))
            virial = np.zeros((3, 3))
            masses = np.array([atomic_masses[n] for n in geometry.per_atom.numbers])
            plumed_.cmd("setStep", 0)
            plumed_.cmd("setPositions", geometry.per_atom.positions.astype(np.float64))
            plumed_.cmd("setMasses", masses)
            plumed_.cmd("setForces", forces)
            plumed_.cmd("setVirial", virial)
            plumed_.cmd("prepareCalc")
            plumed_.cmd("performCalcNoUpdate")
            plumed_.cmd("getBias", energy)
            if geometry.periodic:
                stress = virial / np.linalg.det(geometry.cell)
            else:
                stress = np.zeros((3, 3))

            value[i] = float(energy.item())
            grad_pos[i, :len(geometry)] = forces
            grad_cell[i] = stress
        return {'energy': value, 'forces': grad_pos, 'stress': grad_cell}


@typeguard.typechecked
class ZeroFunction(EnergyFunction):

    def __call__(
        self,
        geometries: list[Geometry],
    ) -> dict[str, np.ndarray]:
        energy, forces, stress = create_outputs(
            self.outputs,
            geometries,
        )
        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue
            energy[i] = 0.0
            forces[i, :len(geometry)] = 0.0
            stress[i, :] = 0.0
        return {'energy': energy, 'forces': forces, 'stress': stress}


@typeguard.typechecked
@dataclass
class HarmonicFunction(EnergyFunction):
    positions: np.ndarray
    hessian: np.ndarray
    energy: Optional[np.ndarray] = None

    def __call__(self, geometries: list[Geometry]) -> dict[str, np.ndarray]:
        energy, forces, stress = create_outputs(self.outputs, geometries)

        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue
            delta = geometry.per_atom.positions.reshape(-1) - self.positions.reshape(-1)
            grad = np.dot(self.hessian, delta)
            energy[i] = 0.5 * np.dot(delta, grad)
            if self.energy is not None:
                energy[i] += self.energy
            forces[i] = (-1.0) * grad.reshape(-1, 3)
            stress[i] = 0.0
        return {'energy': energy, 'forces': forces, 'stress': stress}
