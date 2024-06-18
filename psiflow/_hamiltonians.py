from functools import partial
from typing import Optional

import typeguard
import numpy as np
from parsl.app.app import python_app

import psiflow
from psiflow.function import Function, _apply
from psiflow.geometry import Geometry, NullState
from psiflow.data import create_outputs


@typeguard.typechecked
@psiflow.serializable
class Hamiltonian:
    pass


@typeguard.typechecked
class Harmonic(Hamiltonian):

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


class MACEHamiltonian(Hamiltonian):
    pass


def get_mace_mp0():
    pass


class PlumedHamiltonian(Hamiltonian):
    pass
