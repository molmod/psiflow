from functools import partial

import typeguard
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture
from typing import Optional, Union

import psiflow
from psiflow.function import Function, _apply, EinsteinCrystalFunction
from psiflow.geometry import Geometry
from psiflow.data import Dataset


insert_as = {
    'value': 'energy',
    'grad_pos': 'forces',
    'grad_cell': 'stress',
}


@typeguard.typechecked
@psiflow.serializable
class Hamiltonian:
    function: Function

    def compute(
        self,
        arg: Union[Dataset, AppFuture[list], list[Union[AppFuture, Geometry]]],
        outputs: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
    ) -> Union[list[AppFuture], AppFuture]:
        if outputs is None:
            outputs = ['energy', 'forces', 'stress']
        else:
            outputs = list(outputs)
        inverse = {v: k for k, v in insert_as.items()}
        func_outputs = [inverse[o] for o in outputs]
        return self.function.compute(arg, func_outputs, batch_size)


@typeguard.typechecked
@psiflow.serializable
class EinsteinCrystal(Hamiltonian):
    function: Function

    def __init__(self, **kwargs):
        self.function = EinsteinCrystalFunction(**kwargs)
        self._create_apps()

    def _create_apps(self):
        func = partial(
            _apply,
            func=self.function.apply,
            insert_as=insert_as,
        )
        self.apply_app = python_app(func, executors=['default_threads'])
