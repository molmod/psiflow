from __future__ import annotations  # necessary for type-guarding class methods

from typing import Optional

import typeguard

from psiflow.data import Dataset
from psiflow.hamiltonians.hamiltonian import Hamiltonian
from psiflow.sampling import SimulationOutput, Walker
from psiflow.sampling.walker import quench, randomize


@typeguard.typechecked
class ThermodynamicIntegration:
    hamiltonian0: Hamiltonian
    hamiltonian1: Hamiltonian
    alphas: list[float]
    temperatures: list[float]

    def __init__(
        self,
        hamiltonian0: Hamiltonian,
        hamiltonian1: Hamiltonian,
        alphas: Optional[list[float]] = None,
        temperatures: Optional[list[float]] = None,
    ):
        self.hamiltonian0 = hamiltonian0
        self.hamiltonian1 = hamiltonian1
        if alphas is None:
            alphas = []
        self.alphas = alphas
        if temperatures is None:
            temperatures = []
        self.temperatures = temperatures
        self.walkers = None

    def generate_walkers(
        self,
        dataset: Dataset,
        initialize_by: str = "quench",
        **walker_kwargs,
    ) -> list[Walker]:
        mtd = walker_kwargs.pop("metadynamics", None)
        assert mtd is None  # no time-dependent bias contributions
        if len(self.temperatures) == 0:
            T = walker_kwargs.pop("temperature", None)
            assert T is not None
            self.temperatures = [T]

        walkers = []
        for alpha in self.alphas:
            for T in self.temperatures:
                hamiltonian = (
                    alpha * self.hamiltonian1 + (1 - alpha) * self.hamiltonian0
                )
                walker = Walker(
                    dataset[0],  # do quench later
                    hamiltonian,
                    temperature=T,
                    pressure=walker_kwargs.pop("pressure", None),
                    timestep=walker_kwargs.pop("timestep", 0.5),
                    order_parameter=walker_kwargs.pop("order_parameter", None),
                )
                walkers.append(walker)

        # initialize walkers
        if initialize_by == "quench":
            quench(walkers, dataset)
        elif initialize_by == "shuffle":
            randomize(walkers, dataset)
        else:
            raise ValueError("unknown initialization")

        self.walkers = walkers
        return walkers

    def run(self, steps: int, step: int) -> list[SimulationOutput]:
        assert self.walkers is not None
