from dataclasses import dataclass
from typing import Optional

import typeguard

from psiflow.hamiltonians.hamiltonian import Hamiltonian


@dataclass
@typeguard.typechecked
class Walker:
    hamiltonian: Hamiltonian
    temperature: Optional[float] = 300
    pressure: Optional[float] = None


@dataclass
@typeguard.typechecked
class PathIntegralWalker(Walker):
    nbeads: int = 1
