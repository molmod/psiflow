from __future__ import annotations  # necessary for type-guarding class methods

import os
import logging
import tempfile
from functools import partial
from typing import Union, Optional, Callable
from pathlib import Path

from ase.units import fs, kJ, mol, nm
from ase.data import atomic_masses
import typeguard
import numpy as np
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.app.app import python_app

import psiflow
from psiflow.data import Dataset, batch_apply, _read_frames, _write_frames
from psiflow.geometry import Geometry, NullState, create_outputs

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def _concatenate_multiple(*args: list[np.ndarray]) -> list[np.ndarray]:
    narrays = len(args[0])
    for arg in args:
        assert isinstance(arg, list)
    assert all([len(a) == narrays for a in args])

    concatenated = []
    for i in range(narrays):
        concatenated.append(np.concatenate([arg[i] for arg in args]))
    return concatenated


concatenate_multiple = python_app(_concatenate_multiple, executors=['default_threads'])


def _get_length(arg):
    if isinstance(arg, list):
        return len(arg)
    else:
        return 1


get_length = python_app(_get_length, executors=['default_threads'])


def _apply(
    arg: Union[Geometry, list[Geometry], None],
    insert: bool,
    func_outputs: tuple[str, ...],
    inputs: list = [],
    outputs: list = [],
    func: Optional[Callable] = None,
    **parameters,
) -> Optional[list[np.ndarray]]:
    assert func is not None  # has to be kwarg to support partial()
    if arg is None:
        states = _read_frames(inputs=[inputs[0]])
    elif not isinstance(arg, list):
        states = [arg]
    else:
        states = arg
    output_arrays = func(
        states,
        insert=insert,
        outputs=func_outputs,
        **parameters,
    )
    if len(outputs) > 0:
        _write_frames(*states, outputs=[outputs[0]])
    return output_arrays


@typeguard.typechecked
@psiflow.serializable
class Function:

    def compute(
        self,
        arg: Union[Dataset, AppFuture[list], list[Union[AppFuture, Geometry]]],
        outputs: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
    ) -> Union[list[AppFuture], AppFuture]:
        if outputs is None:
            func_outputs = list(self.outputs)
        else:
            for output in outputs:
                assert output in self.outputs
            func_outputs = list(outputs)
        if batch_size is not None:
            if isinstance(arg, Dataset):
                length = arg.length()
            else:
                length = get_length(arg)
                # convert to Dataset for convenience
                arg = Dataset(arg)
            future = batch_apply(
                self.apply_app,
                arg,
                batch_size,
                length,
                reduce_func=concatenate_multiple,  # concatenate arrays
                func_outputs=func_outputs,
                **self.parameters(),
            )
        else:
            if isinstance(arg, Dataset):
                future = self.apply_app(
                    None,
                    insert=False,
                    func_outputs=func_outputs,
                    inputs=[arg.extxyz],
                    **self.parameters(),
                )
            else:
                future = self.apply_app(
                    arg,
                    insert=False,
                    func_outputs=func_outputs,
                    inputs=[],
                    **self.parameters(),
                )
        if len(func_outputs) == 1:
            return future[0]
        else:
            return [future[i] for i in range(len(func_outputs))]

    @staticmethod
    def apply(
        geometries: list[Geometry],
        outputs: Optional[list[str]] = None,
        insert: bool = False,
        **kwargs,
    ) -> list[np.ndarray]:
        raise NotImplementedError

    def parameters(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def set_outputs(
        geometries: list[Geometry],
        **kwargs,
    ):
        for i, geometry in enumerate(geometries):
            for name, array in kwargs.items():
                if name == 'energy':
                    geometry.energy = array[i]
                elif name == 'forces':
                    geometry.per_atom.forces[:] = array[i, :len(geometry)]
                elif name == 'stress':
                    geometry.stress[:] = array[i]
                else:
                    raise ValueError

    @staticmethod
    def sort_outputs(
        outputs: list[str],
        **kwargs,
    ) -> list[np.ndarray]:
        output_arrays = []
        for name in outputs:
            array = kwargs.get(name, None)
            assert array is not None
            output_arrays.append(array)
        return output_arrays


@typeguard.typechecked
@psiflow.serializable
class EinsteinCrystalFunction(Function):
    outputs = ('energy', 'forces', 'stress')
    force_constant: float
    centers: np.ndarray
    volume: float

    def __init__(
        self,
        force_constant: float,
        centers: np.ndarray,
        volume: float = 0.0,
        **kwargs,
    ):
        self.force_constant = force_constant
        self.centers = centers
        self.volume = volume
        self._create_apps()

    def _create_apps(self):
        func = partial(
            _apply,
            func=EinsteinCrystalFunction.apply,
        )
        self.apply_app = python_app(func, executors=['default_threads'])

    def parameters(self):
        return {
            'force_constant': self.force_constant,
            'centers': self.centers,
            'volume': self.volume,
        }

    def apply(
        geometries: list[Geometry],
        force_constant: float,
        centers: np.ndarray,
        volume: float = 0.0,
        insert: bool = False,
        outputs: Optional[list[str]] = None,
    ) -> list[np.ndarray]:
        energy, forces, stress = create_outputs(
            EinsteinCrystalFunction.outputs,
            geometries,
        )

        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue

            delta = geometry.per_atom.positions - centers
            energy[i] = force_constant * np.sum(delta ** 2) / 2
            forces[i, :len(geometry)] = (-1.0) * force_constant * delta
            if geometry.periodic and volume > 0.0:
                delta = np.linalg.det(geometry.cell) - volume
                _stress = force_constant * np.eye(3) * delta
            else:
                _stress = np.zeros((3, 3))
            stress[i, :] = _stress

        if outputs is None:
            outputs = EinsteinCrystalFunction.outputs
        output_arrays = Function.sort_outputs(
            outputs,
            energy=energy,
            forces=forces,
            stress=stress,
        )
        if insert:
            Function.set_outputs(
                geometries,
                **{k: v for k, v in zip(outputs, output_arrays)},
            )
        return output_arrays


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
