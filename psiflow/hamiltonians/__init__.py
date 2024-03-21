import json

from ase.calculators.calculator import Calculator

from ._einstein import EinsteinCrystal  # noqa: F401
from ._mace import MACEHamiltonian  # noqa: F401
from ._plumed import PlumedHamiltonian  # noqa: F401
from .hamiltonian import Hamiltonian  # noqa: F401


def deserialize(path) -> Calculator:
    with open(path, "r") as f:
        data = json.loads(f.read())
    assert "hamiltonian" in data
    hamiltonian_cls = globals()[data.pop("hamiltonian")]
    return hamiltonian_cls.deserialize(**data)
