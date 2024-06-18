from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import Union, Optional, Callable
from functools import partial

import typeguard
import numpy as np
from parsl.dataflow.futures import AppFuture
from parsl.app.app import python_app

import psiflow
from psiflow.data import Dataset, batch_apply, _read_frames, _write_frames, create_outputs
from psiflow.geometry import Geometry, NullState

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


@staticmethod
def _sort_outputs(
    outputs: list[str],
    **kwargs,
) -> list[np.ndarray]:
    output_arrays = []
    for name in outputs:
        array = kwargs.get(name, None)
        assert array is not None
        output_arrays.append(array)
    return output_arrays


sort_outputs = python_app(_sort_outputs, executors=['default_threads'])


def _insert_outputs(
    geometries: list[Geometry],
    outputs_: list[str],
    output_arrays: list[np.ndarray],
    insert_as: dict[str, str],
) -> None:
    for i, geometry in enumerate(geometries):
        for name, array in zip(outputs_, output_arrays):
            key = insert_as.get(name, None)
            if key is not None:
                if key == 'energy':
                    geometry.energy = array[i]
                elif key == 'forces':
                    geometry.per_atom.forces[:] = array[i, :len(geometry)]
                elif key == 'stress':
                    geometry.stress[:] = array[i]
                else:
                    raise ValueError


insert_outputs = python_app(_insert_outputs, executors=['default_threads'])


def _apply(
    arg: Union[Geometry, list[Geometry], None],
    outputs_: tuple[str, ...],
    inputs: list = [],
    outputs: list = [],
    func: Optional[Callable] = None,
    insert_as: Optional[dict[str, str]] = None,
    **parameters,
) -> Optional[list[np.ndarray]]:
    assert func is not None  # has to be kwarg to support partial()
    if arg is None:
        states = _read_frames(inputs=[inputs[0]])
    elif not isinstance(arg, list):
        states = [arg]
    else:
        states = arg
    function = Function(**parameters)
    output_dict = function(states)
    output_arrays = _sort_outputs()
    if insert_as is not None:
        insert_outputs(
            states,
            outputs_,
            output_arrays,
            insert_as=insert_as,
        )
    if len(outputs) > 0:
        _write_frames(*states, outputs=[outputs[0]])
    return output_arrays

def compute(
    self,
    arg: Union[Dataset, AppFuture[list], list[Union[AppFuture, Geometry]]],
    outputs: Optional[list[str]] = None,
    batch_size: Optional[int] = None,
) -> Union[list[AppFuture], AppFuture]:
    if outputs is None:
        outputs = list(self.outputs)
    else:
        for output in outputs:
            assert output in self.outputs
        outputs = list(outputs)
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
            outputs_=outputs,
            **self.parameters(),
        )
    else:
        if isinstance(arg, Dataset):
            future = self.apply_app(
                None,
                outputs_=outputs,
                inputs=[arg.extxyz],
                **self.parameters(),
            )
        else:
            future = self.apply_app(
                arg,
                outputs_=outputs,
                inputs=[],
                **self.parameters(),
            )
    if len(outputs) == 1:
        return future[0]
    else:
        return [future[i] for i in range(len(outputs))]


@typeguard.typechecked
class Function:
    outputs = ('value', 'grad_pos', 'grad_cell')

    def __call__(
        self,
        geometries: list[Geometry],
    ) -> list[np.ndarray]:
        raise NotImplementedError

    def parameters(self) -> dict:
        raise NotImplementedError


# @typeguard.typechecked
# @psiflow.serializable
class EinsteinCrystalFunction(Function):
    outputs = ('value', 'grad_pos', 'grad_cell')
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

    #def _create_apps(self):
    #    func = partial(
    #        _apply,
    #        func=EinsteinCrystalFunction.apply,
    #    )
    #    self.apply_app = python_app(func, executors=['default_threads'])

    def parameters(self):
        return {
            'force_constant': self.force_constant,
            'centers': self.centers,
            'volume': self.volume,
        }

    @staticmethod
    def __call__(
        self,
        geometries: list[Geometry],
    ) -> list[np.ndarray]:
        value, grad_pos, grad_cell = create_outputs(
            ['energy', 'forces', 'stress'],
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

        return {'value': value, 'grad_pos': grad_pos, 'grad_cell': grad_cell}


# @typeguard.typechecked
# class CustomFunction(Function):
#     outputs: tuple = ('energy', 'forces', 'stress')
