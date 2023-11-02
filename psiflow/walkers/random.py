from __future__ import annotations  # necessary for type-guarding class methods

from collections import namedtuple
from typing import Any, Union

import numpy as np
import typeguard
from ase import Atoms
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms
from psiflow.utils import unpack_i
from psiflow.walkers import BaseWalker

Metadata = namedtuple("Metadata", ["state", "counter", "reset"])


def apply_strain(strain, box0):
    """Applies a strain tensor to a reference box

    The resulting strained box matrix is obtained based on:

        box = box0 @ sqrt(2 * strain + I)

    where the second argument is computed based on a diagonalization of
    2 * strain + I.

    Parameters
    ----------

    strain : ndarray of shape (3, 3)
        desired strain matrix

    box0 : ndarray of shape (3, 3)
        reference box matrix

    """
    assert np.allclose(strain, strain.T)
    A = 2 * strain + np.eye(3)
    values, vectors = np.linalg.eigh(A)
    sqrtA = vectors @ np.sqrt(np.diag(values)) @ vectors.T
    box = box0 @ sqrtA
    return box


def compute_strain(box, box0):
    """Computes the strain of a given box with respect to a reference

    The strain matrix is defined by the following expression

        strain = 0.5 * (inv(box0) @ box @ box.T @ inv(box0).T - I)

    Parameters
    ----------

    box : ndarray of shape (3, 3)
        box matrix for which to compute the strain

    box0 : ndarray of shape (3, 3)
        reference box matrix

    """
    box0inv = np.linalg.inv(box0)
    return 0.5 * (box0inv @ box @ box.T @ box0inv.T - np.eye(3))


@typeguard.typechecked
def random_perturbation(
    state: FlowAtoms,
    parameters: dict[str, Any],
) -> tuple[FlowAtoms, int, bool]:
    import copy

    import numpy as np

    from psiflow.walkers.random import apply_strain

    state = copy.deepcopy(state)
    np.random.seed(parameters["seed"])
    if parameters["amplitude_box"] > 0:
        assert state.pbc.all()
        frac = state.positions @ np.linalg.inv(state.cell)
        strain = np.random.uniform(
            -parameters["amplitude_box"],
            parameters["amplitude_box"],
            size=(3, 3),
        )
        strain[0, 1] = strain[1, 0]  # strain is symmetric
        strain[0, 2] = strain[2, 0]
        strain[1, 2] = strain[2, 1]
        box = apply_strain(strain, state.cell)
        positions = frac @ box
        state.set_cell(box)
    else:
        positions = state.positions
    positions += np.random.uniform(
        -parameters["amplitude_pos"],
        parameters["amplitude_pos"],
        size=state.positions.shape,
    )
    state.set_positions(positions)
    return state, 1, False


app_random_perturbation = python_app(random_perturbation, executors=["default_threads"])


@typeguard.typechecked
class RandomWalker(BaseWalker):
    def __init__(
        self,
        atoms: Union[Atoms, FlowAtoms, AppFuture],
        amplitude_pos=0.1,
        amplitude_box=0.0,
        **kwargs,
    ) -> None:
        super().__init__(atoms, **kwargs)
        self.amplitude_pos = amplitude_pos
        self.amplitude_box = amplitude_box

    def _propagate(self, **kwargs):
        result = app_random_perturbation(
            self.state,
            self.parameters,
        )
        metadata = Metadata(*[unpack_i(result, i) for i in range(3)])
        return metadata, None  # no output trajectory

    @property
    def parameters(self) -> dict[str, Any]:
        parameters = super().parameters
        parameters["amplitude_pos"] = self.amplitude_pos
        parameters["amplitude_box"] = self.amplitude_box
        return parameters

    @classmethod
    def create_apps(cls) -> None:
        pass
