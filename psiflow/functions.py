import json
import os
import tempfile
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from collections.abc import Sequence

import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_masses
from ase.units import fs, kJ, mol, nm
from ase.stress import voigt_6_to_full_3x3_stress

from psiflow.geometry import Geometry


def format_output(
    geometry: Geometry,
    energy: float,
    forces: np.ndarray | None = None,
    stress: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """"""
    forces = np.zeros(shape=(len(geometry), 3)) if forces is None else forces
    stress = np.zeros(shape=(3, 3)) if stress is None else stress
    if stress.size == 6:
        stress = voigt_6_to_full_3x3_stress(stress)
    return {"energy": energy, "forces": forces, "stress": stress}


class Function:
    outputs: tuple = ("energy", "forces", "stress")

    def __call__(
        self,
        geometry: Geometry,
    ) -> dict[str, float | np.ndarray]:
        raise NotImplementedError

    def compute(
        self,
        geometries: Sequence[Geometry],
    ) -> dict[str, list[float | np.ndarray]]:
        """Evaluate multiple geometries and merge data into single arrays"""
        data = {k: [] for k in self.outputs}
        for i, geometry in enumerate(geometries):
            out = self(geometry)
            for k, v in out.items():
                data[k].append(v)
        return data


@dataclass(frozen=True)
class EinsteinCrystalFunction(Function):
    force_constant: float
    centers: np.ndarray
    volume: float = 0.0

    def __call__(
        self,
        geometry: Geometry,
    ) -> dict[str, float | np.ndarray]:
        delta = geometry.per_atom.positions - self.centers
        energy = self.force_constant * np.sum(delta**2) / 2
        grad_pos = (-1.0) * self.force_constant * delta
        grad_cell = None
        if geometry.periodic and self.volume > 0.0:
            delta = np.linalg.det(geometry.cell) - self.volume
            grad_cell = self.force_constant * np.eye(3) * delta
        return format_output(geometry, energy, grad_pos, grad_cell)


@dataclass(frozen=True)
class PlumedFunction(Function):
    plumed_input: str
    external: Optional[Union[str, Path]] = None
    plumed_instances: dict[tuple, "plumed.Plumed"] = field(default_factory=dict)

    def __post_init__(self):
        if self.external is not None:
            assert self.external in self.plumed_input

    def __call__(
        self,
        geometry: Geometry,
    ) -> dict[str, float | np.ndarray]:
        if not geometry.periodic and "VOLUME" in self.plumed_input:
            raise ValueError("VOLUME CV only supported for periodic structures")

        key = self._geometry_to_key(geometry)
        if key not in self.plumed_instances:
            from plumed import Plumed

            tmp = tempfile.NamedTemporaryFile(prefix="plumed_", mode="w+", delete=False)
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
            for line in self.plumed_input.split("\n"):
                plumed_.cmd("readInputLine", line)
            os.remove(tmp.name)  # remove whatever plumed has created
            self.plumed_instances[key] = plumed_

        # input system
        plumed_ = self.plumed_instances[key]
        plumed_.cmd("setStep", 0)
        masses = np.array([atomic_masses[n] for n in geometry.per_atom.numbers])
        plumed_.cmd("setMasses", masses)
        copied_positions = geometry.per_atom.positions.astype(np.float64, copy=True)
        plumed_.cmd("setPositions", copied_positions)
        if geometry.periodic:
            cell = geometry.cell.astype(np.float64, copy=True)
            plumed_.cmd("setBox", cell)

        # perform calculation
        energy = np.zeros((1,))
        forces = np.zeros((len(geometry), 3))
        virial = np.zeros((3, 3))
        plumed_.cmd("setForces", forces)
        plumed_.cmd("setVirial", virial)
        plumed_.cmd("prepareCalc")
        plumed_.cmd("performCalcNoUpdate")
        plumed_.cmd("getBias", energy)
        stress = None
        if geometry.periodic:
            stress = virial / np.linalg.det(geometry.cell)
        return format_output(geometry, float(energy.item()), forces, stress)

    @staticmethod
    def _geometry_to_key(geometry: Geometry) -> tuple:
        return tuple([geometry.periodic]) + tuple(geometry.numbers)


@dataclass(frozen=True)
class ZeroFunction(Function):
    def __call__(
        self,
        geometry: Geometry,
    ) -> dict[str, float | np.ndarray]:
        return format_output(geometry, 0.0)


@dataclass(frozen=True)
class HarmonicFunction(Function):
    positions: np.ndarray
    hessian: np.ndarray
    energy: Optional[float] = None

    def __call__(
        self,
        geometry: Geometry,
    ) -> dict[str, float | np.ndarray]:
        delta = geometry.per_atom.positions.reshape(-1) - self.positions.reshape(-1)
        grad = np.dot(self.hessian, delta)
        energy = 0.5 * np.dot(delta, grad)
        if self.energy is not None:
            energy += self.energy
        return format_output(geometry, energy, (-1.0) * grad.reshape(-1, 3))


@dataclass(frozen=True)
class MACEFunction(Function):
    # TODO: why are some arguments separated and others 'calc_kwargs'?
    model_path: str | Path
    ncores: int
    device: str
    dtype: str
    calc_kwargs: dict = field(default_factory=dict)
    env_vars: Optional[dict[str, str]] = None

    calc: Calculator = field(init=False)

    def __post_init__(self):
        # import environment variables before any nontrivial imports
        if self.env_vars is not None:
            for key, value in self.env_vars.items():
                os.environ[key] = value

        import torch
        from mace.calculators.mace import MACECalculator

        # MACE uses the root logger..
        logging.getLogger("").setLevel(logging.INFO)

        torch.set_num_threads(self.ncores)
        calc = MACECalculator(
            model_paths=self.model_path,
            device=self.device,
            default_dtype=self.dtype,
            **self.calc_kwargs,
        )
        object.__setattr__(self, "calc", calc)  # frozen dataclass instance

    def __call__(
        self,
        geometry: Geometry,
    ) -> dict[str, float | np.ndarray]:
        atoms = geometry.to_atoms(structural_only=True)
        self.calc.calculate(atoms)
        return format_output(geometry, **self.calc.results)


@dataclass(frozen=True)
class DispersionFunction(Function):
    method: str
    damping: str = "d3bj"
    params_tweaks: Optional[dict[str, float]] = None
    num_threads: int = 4

    calc: Calculator = field(init=False)

    def __post_init__(self):
        # OMP_NUM_THREADS for parallel evaluation does not work..
        # https://github.com/dftd3/simple-dftd3/issues/49
        # TODO: check whether this is still the case
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

        from dftd3.ase import DFTD3

        calc = DFTD3(
            method=self.method, damping=self.damping, params_tweaks=self.params_tweaks
        )
        object.__setattr__(self, "calc", calc)  # frozen dataclass instance

    def __call__(
        self,
        geometry: Geometry,
    ) -> dict[str, float | np.ndarray]:
        atoms = geometry.to_atoms(structural_only=True)
        self.calc.calculate(atoms)
        return format_output(geometry, **self.calc.results)





def function_from_json(path: Union[str, Path], **kwargs) -> Function:
    from psiflow.serialization import deserialize_hook

    functions = [
        EinsteinCrystalFunction,
        HarmonicFunction,
        MACEFunction,
        PlumedFunction,
        DispersionFunction,
    ]
    with open(path, "r") as f:
        data = json.loads(f.read(), object_hook=deserialize_hook)
    function_name = data.pop("function_name")
    function_cls = {f.__name__: f for f in functions}[function_name]

    for key, value in kwargs.items():
        if key in data:
            data[key] = value
    function = function_cls(**data)
    return function
