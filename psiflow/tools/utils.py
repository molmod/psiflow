from __future__ import annotations  # necessary for type-guarding class methods

import numpy as np
import typeguard
from ase.data import atomic_masses
from parsl.app.app import python_app

from psiflow.data import Geometry


@typeguard.typechecked
def get_mass_matrix(geometry: Geometry) -> np.ndarray:
    masses = np.repeat(
        np.array([atomic_masses[n] for n in geometry.per_atom.numbers]),
        3,
    )
    sqrt_inv = 1 / np.sqrt(masses)
    return np.outer(sqrt_inv, sqrt_inv)


@typeguard.typechecked
def _mass_weight(hessian: np.ndarray, geometry: Geometry) -> np.ndarray:
    assert hessian.shape[0] == hessian.shape[1]
    assert len(geometry) * 3 == hessian.shape[0]
    return hessian * get_mass_matrix(geometry)


mass_weight = python_app(_mass_weight, executors=["default_threads"])


@typeguard.typechecked
def _mass_unweight(hessian: np.ndarray, geometry: Geometry) -> np.ndarray:
    assert hessian.shape[0] == hessian.shape[1]
    assert len(geometry) * 3 == hessian.shape[0]
    return hessian / get_mass_matrix(geometry)


mass_unweight = python_app(_mass_unweight, executors=["default_threads"])


@typeguard.typechecked
def _compute_frequencies(hessian: np.ndarray, geometry: Geometry) -> np.ndarray:
    assert hessian.shape[0] == hessian.shape[1]
    assert len(geometry) * 3 == hessian.shape[0]
    return np.sqrt(np.linalg.eigvalsh(_mass_weight(hessian, geometry))) / (2 * np.pi)


compute_frequencies = python_app(_compute_frequencies, executors=["default_threads"])
