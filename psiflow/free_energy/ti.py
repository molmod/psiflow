from __future__ import annotations  # necessary for type-guarding class methods

from typing import Optional

import numpy as np
import typeguard
from parsl.app.app import python_app

from psiflow.data import Dataset
from psiflow.hamiltonians.hamiltonian import Hamiltonian, Zero
from psiflow.sampling import SimulationOutput, Walker, sample
from psiflow.sampling.walker import quench, randomize


@typeguard.typechecked
class ThermodynamicState:
    temperature: float
    delta: Optional[Hamiltonian]
    scale: Optional[float]
    pressure: Optional[float]
    mass: Optional[float]

    def __init__(
        self,
        temperature: float,
        delta: Optional[Hamiltonian],
        scale: Optional[float],
        pressure: Optional[float],
        mass: Optional[float],
    ):
        self.temperature = temperature
        self.delta = delta
        self.scale = scale
        self.pressure = pressure
        self.mass = mass

        self.gradients = {
            "temperature": None,
            "lambda": None,
            "pressure": None,
            "mass": None,
        }

    def gradient(
        self,
        output: SimulationOutput,
        hamiltonian: Optional[Hamiltonian] = None,
    ):
        self.temperature_gradient(output, hamiltonian)
        self.lambda_gradient(output)
        if self.pressure is not None:
            self.pressure_gradient(output)
        if self.mass is not None:
            self.mass_gradient(output)

    def temperature_gradient(
        self,
        output: SimulationOutput,
        hamiltonian: Optional[Hamiltonian] = None,
    ):
        pass

    def lambda_gradient(self, output: SimulationOutput):
        energies = output.get_energy(self.delta)
        take_mean = python_app(np.mean, executors=["default_threads"])
        self.gradients["lambda"] = take_mean(energies)

    def pressure_gradient(output):
        raise NotImplementedError

    def mass_gradient(output):
        raise NotImplementedError


@typeguard.typechecked
class Integration:
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        temperatures: list[float],
        delta: Optional[Hamiltonian],
        npoints: Optional[int],
    ):
        self.hamiltonian = hamiltonian
        self.temperatures = temperatures
        if delta is not None:
            assert npoints is not None
            self.delta = delta
            self.scales = np.linspace(0, 1, num=npoints, endpoint=True)
        else:
            self.scales = np.array([1.0])
            self.delta = Zero()

        self.states = []
        self.walkers = []
        self.outputs = []

    def create_walkers(
        self,
        dataset: Dataset,
        initialize_by: str = "quench",
        **walker_kwargs,
    ) -> list[Walker]:
        for scale in self.scales:
            for T in self.temperatures:
                hamiltonian = self.hamiltonian + scale * self.delta
                walker = Walker(
                    dataset[0],  # do quench later
                    hamiltonian,
                    temperature=T,
                    **walker_kwargs,
                )
                self.walkers.append(walker)
                state = ThermodynamicState(
                    temperature=T,
                    delta=self.delta,
                    scale=scale,
                    pressure=None,
                    mass=None,
                )
                self.states.append(state)

        # initialize walkers
        if initialize_by == "quench":
            quench(self.walkers, dataset)
        elif initialize_by == "shuffle":
            randomize(self.walkers, dataset)
        else:
            raise ValueError("unknown initialization")
        return self.walkers

    def sample(self, **sampling_kwargs):
        self.outputs[:] = sample(
            self.walkers,
            **sampling_kwargs,
        )

    def compute_gradients(self):
        for output, state in zip(self.outputs, self.states):
            state.gradient(output, hamiltonian=self.hamiltonian)
