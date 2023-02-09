from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Tuple
import typeguard
from dataclasses import dataclass

from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms
from psiflow.execution import ExecutionContext
from psiflow.sampling import BaseWalker, PlumedBias
from psiflow.models import BaseModel


@typeguard.typechecked
def random_perturbation(
        state: FlowAtoms,
        parameters: RandomParameters,
        ) -> Tuple[FlowAtoms, str, int]:
    import numpy as np
    from psiflow.sampling.utils import apply_strain
    np.random.seed(parameters.seed)
    frac = state.positions @ np.linalg.inv(state.cell)
    strain = np.random.uniform(
            -parameters.amplitude_box,
            parameters.amplitude_box,
            size=(3, 3),
            )
    strain[0, 1] = strain[1, 0] # strain is symmetric
    strain[0, 2] = strain[2, 0]
    strain[1, 2] = strain[2, 1]
    box = apply_strain(strain, state.cell)
    positions = frac @ box
    positions += np.random.uniform(
            -parameters.amplitude_pos,
            parameters.amplitude_pos,
            size=state.positions.shape,
            )
    state.set_positions(positions)
    state.set_cell(box)
    return state, 'safe', 1
app_propagate = python_app(random_perturbation, executors=['default'])


@typeguard.typechecked
def propagate_wrapped(
        state: AppFuture,
        parameters: RandomParameters,
        keep_trajectory: bool = False,
        **kwargs,
        ) -> AppFuture:
    # ignore additional kwargs; return None as dataset
    assert not keep_trajectory
    future = app_propagate(state, parameters)
    return future


@dataclass
class RandomParameters:
    amplitude_pos: float = 0.05
    amplitude_box: float = 0.05
    seed         : int = 0


@typeguard.typechecked
class RandomWalker(BaseWalker):
    parameters_cls = RandomParameters

    def get_propagate_app(self, model):
        return propagate_wrapped

    @classmethod
    def create_apps(cls, context: ExecutionContext) -> None:
        pass
