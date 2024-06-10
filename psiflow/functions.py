from __future__ import annotations  # necessary for type-guarding class methods

import os
import logging
import tempfile
from typing import Union, Type
from pathlib import Path

from ase.units import fs, kJ, mol, nm
from ase.data import atomic_masses
import numpy as np
import typeguard
from parsl.data_provider.files import File

from psiflow.geometry import Geometry, NullState

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def apply_function(
    function_cls: Type[Function],
    geometries: Union[Geometry, list[Geometry]],
    **parameters,
) -> dict[str, np.ndarray]:

    if isinstance(geometries, Geometry):
        geometries = [geometries]

    function = function_cls(**parameters)
    outputs = function(geometries)
    return outputs


@typeguard.typechecked
def read_apply(
    function_cls: Type[Function],
    inputs: list = [],
    parsl_resource_specification: dict = {},
    **parameters,  # dict values can be futures, so app must wait for those
) -> list[np.ndarray]:
    from psiflow.data import _read_frames
    geometries = _read_frames(None, inputs=[inputs[0]])
    return apply_function(function_cls, geometries, **parameters)


@typeguard.typechecked
class Function:
    outputs: tuple = ()

    def __call__(self, geometries: list[Geometry]) -> dict[str, np.ndarray]:
        pass

    def create_outputs(self, geometries: list[Geometry]):
        nframes = len(geometries)
        natoms = [len(g) for g in geometries]
        max_natoms = np.max(natoms)

        output_names = list(self.outputs)
        outputs = {}

        for key in list(self.outputs):
            output_names.remove(key)
            if key == 'energy':
                array = np.empty((nframes,), dtype=np.float32)
            elif key == 'forces':
                array = np.empty((nframes, max_natoms, 3), dtype=np.float32)
            elif key == 'stress':
                array = np.empty((nframes, 3, 3), dtype=np.float32)
            else:
                raise ValueError
            array[:] = np.nan
            outputs[key] = array

        return outputs


@typeguard.typechecked
class EinsteinCrystalFunction(Function):
    outputs: tuple = ('energy', 'forces', 'stress')

    def __init__(self, force_constant: float, centers: np.ndarray, volume: float):
        self.force_constant = force_constant
        self.centers = centers
        self.volume = volume

    def __call__(self, geometries: list[Geometry]) -> dict[str, np.ndarray]:
        outputs = self.create_outputs(geometries)

        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue
            delta = geometry.per_atom.positions - self.centers
            outputs['energy'][i] = self.force_constant * np.sum(delta ** 2) / 2
            outputs['forces'][i, :len(geometry)] = (-1.0) * self.force_constant * delta
            if geometry.periodic and self.volume > 0.0:
                delta = np.linalg.det(geometry.cell) - self.volume
                outputs['stress'][i] = self.force_constant * np.eye(3) * delta
            else:
                outputs['stress'][i] = 0.0
        return outputs


@typeguard.typechecked
class PlumedFunction(Function):
    outputs: tuple = ('energy', 'forces', 'stress')

    def __init__(
        self,
        plumed_input: str,
        external: Union[str, Path, File, None] = None,
    ):
        self.plumed_input = plumed_input
        if external is not None:
            if isinstance(external, File):
                external = external.filepath
            assert Path(external).exists()
        self.external = external

    def __call__(self, geometries):
        outputs = self.create_outputs(geometries)

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

            outputs['energy'][i] = float(energy.item())
            outputs['forces'][i, :len(geometry)] = forces
            outputs['stress'][i] = stress
        return outputs


@typeguard.typechecked
class HarmonicFunction(Function):
    outputs: tuple = ('energy', 'forces', 'stress')

    def __init__(
        self,
        positions: np.ndarray,
        hessian: np.ndarray,
        energy: float,
    ):
        self.positions = positions
        self.hessian = hessian
        self.energy = energy

    def __call__(self, geometries: list[Geometry]) -> dict[str, np.ndarray]:
        outputs = self.create_outputs(geometries)

        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue
            delta = geometry.per_atom.positions.reshape(-1) - self.positions.reshape(-1)
            grad = np.dot(self.hessian, delta)
            outputs['energy'][i] = self.energy + 0.5 * np.dot(delta, grad)
            outputs['forces'][i] = (-1.0) * grad.reshape(-1, 3)
            outputs['stress'][i] = 0.0
        return outputs


@typeguard.typechecked
class CustomFunction(Function):
    outputs: tuple = ('energy', 'forces', 'stress')


def extend_with_future_support(function_cls: Type[Function]):
    print(function_cls.__dict__)
    pass
