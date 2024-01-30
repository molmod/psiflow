import numpy as np
from scipy.stats import chi2


def max_temperature(temperature: float, natoms: int, quantile: float) -> float:
    ndof = 3 * natoms
    return chi2.ppf(1 - quantile, ndof) * temperature / ndof


def max_temperature_std(temperature: float, natoms: int, N: float) -> float:
    return temperature + N * temperature / np.sqrt(3 * natoms)


def get_velocities_at_temperature(temperature, masses):
    from ase.units import kB

    velocities = np.random.normal(0, 1, (len(masses), 3))
    velocities *= np.sqrt(kB * temperature / masses).reshape(-1, 1)
    actual = (velocities**2 * masses.reshape(-1, 1)).mean() / kB
    scale = np.sqrt(temperature / actual)
    velocities *= scale
    return velocities
