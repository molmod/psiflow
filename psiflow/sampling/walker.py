from dataclasses import dataclass
from typing import Optional

import typeguard

from psiflow.hamiltonians import Hamiltonian


@dataclass
@typeguard.typechecked
class Walker:
    hamiltonian: Hamiltonian
    temperature: Optional[float] = 300
    presssure: Optional[float] = None
    steps: int = 100
    step: int = 10
    start: int = 0
    max_excess_temperature: float = 1e6
    distance_threshold: float = 0.5
