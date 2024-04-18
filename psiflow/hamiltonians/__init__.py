import json
from pathlib import Path
from typing import Optional, Union

import typeguard
from ase.calculators.calculator import Calculator

from ._einstein import EinsteinCrystal  # noqa: F401
from ._harmonic import Harmonic  # noqa: F401
from ._mace import MACEHamiltonian, get_mace_mp0  # noqa: F401
from ._plumed import PlumedHamiltonian  # noqa: F401
from .hamiltonian import Hamiltonian  # noqa: F401


@typeguard.typechecked
def deserialize_calculator(
    path: Union[str, Path],
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    max_force: Optional[float] = None,
) -> Calculator:
    with open(path, "r") as f:
        data = json.loads(f.read())
    assert "hamiltonian" in data
    hamiltonian_cls = globals()[data.pop("hamiltonian")]

    if device is not None:
        data["device"] = device
    if dtype is not None:
        data["dtype"] = dtype
    calculator = hamiltonian_cls.deserialize_calculator(**data)
    calculator.max_force = max_force
    return calculator
