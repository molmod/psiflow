from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, Any
import typeguard
from dataclasses import dataclass

from ase import Atoms

from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms
from psiflow.sampling import BaseWalker, PlumedBias
from psiflow.models import BaseModel


@typeguard.typechecked
def random_perturbation(
        state: FlowAtoms,
        parameters: dict[str, Any],
        ) -> tuple[FlowAtoms, str, int]:
    import numpy as np
    import copy
    from psiflow.sampling.utils import apply_strain
    state = copy.deepcopy(state)
    np.random.seed(parameters['seed'])
    frac = state.positions @ np.linalg.inv(state.cell)
    strain = np.random.uniform(
            -parameters['amplitude_box'],
            parameters['amplitude_box'],
            size=(3, 3),
            )
    strain[0, 1] = strain[1, 0] # strain is symmetric
    strain[0, 2] = strain[2, 0]
    strain[1, 2] = strain[2, 1]
    box = apply_strain(strain, state.cell)
    positions = frac @ box
    positions += np.random.uniform(
            -parameters['amplitude_pos'],
            parameters['amplitude_pos'],
            size=state.positions.shape,
            )
    state.set_positions(positions)
    state.set_cell(box)
    return state, 'safe', 1
app_random_perturbation = python_app(random_perturbation, executors=['default'])


@typeguard.typechecked
class RandomWalker(BaseWalker):

    def __init__(self,
            atoms: Union[Atoms, FlowAtoms, AppFuture],
            amplitude_pos=0.1,
            amplitude_box=0.0,
            **kwargs,
            ) -> None:
        super().__init__(atoms, **kwargs)
        self.amplitude_pos = amplitude_pos
        self.amplitude_box = amplitude_box

    def _propagate(self, **kwargs):
        return app_random_perturbation( # no additional kwargs needed
                self.state_future,
                self.parameters,
                )

    @property
    def parameters(self) -> dict[str, Any]:
        parameters = super().parameters
        parameters['amplitude_pos'] = self.amplitude_pos
        parameters['amplitude_box'] = self.amplitude_box
        return parameters

    @classmethod
    def create_apps(cls) -> None:
        pass
