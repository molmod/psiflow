from __future__ import annotations  # necessary for type-guarding class methods

from typing import Optional, Union

import numpy as np
import typeguard
from ase.units import bar, kB
from parsl.app.app import python_app

from psiflow.data import Dataset
from psiflow.hamiltonians.hamiltonian import Hamiltonian, Zero
from psiflow.sampling import SimulationOutput, Walker, sample
from psiflow.sampling.walker import quench, randomize
from psiflow.utils import multiply

length = python_app(len, executors=["default_threads"])
take_mean = python_app(np.mean, executors=["default_threads"])


def _compute_sum(a, b):
    return np.add(a, b)


compute_sum = python_app(_compute_sum, executors=["default_threads"])


@typeguard.typechecked
def _integrate(x: np.ndarray, *args: float) -> np.ndarray:
    import scipy.integrate

    assert len(args) == len(x)
    y = np.array(args, dtype=float)
    return scipy.integrate.cumtrapz(y, x=x, initial=0.0)


integrate = python_app(_integrate, executors=["default_threads"])


@typeguard.typechecked
class ThermodynamicState:
    temperature: float
    natoms: int
    delta_hamiltonian: Optional[Hamiltonian]
    pressure: Optional[float]
    mass: Optional[float]

    def __init__(
        self,
        temperature: float,
        natoms: int,
        delta_hamiltonian: Optional[Hamiltonian],
        pressure: Optional[float],
        mass: Optional[float],
    ):
        self.temperature = temperature
        self.natoms = natoms
        self.delta_hamiltonian = delta_hamiltonian
        self.pressure = pressure
        self.mass = mass

        self.gradients = {
            "temperature": None,
            "delta": None,
            "pressure": None,
            "mass": None,
        }

    def gradient(
        self,
        output: SimulationOutput,
        hamiltonian: Optional[Hamiltonian] = None,
    ):
        self.temperature_gradient(output, hamiltonian)
        self.delta_gradient(output)
        if self.mass is not None:
            self.mass_gradient(output)

    def temperature_gradient(
        self,
        output: SimulationOutput,
        hamiltonian: Optional[Hamiltonian] = None,
    ):
        energies = output.get_energy(hamiltonian)
        _energy = take_mean(energies)
        if self.pressure is not None:  # use enthalpy
            volumes = output["volume{angstrom3}"]
            pv = multiply(take_mean(volumes), 10 * bar * self.pressure)
            _energy = compute_sum(_energy, pv)

        # grad_u = < - u / kBT**2 >
        # grad_k = < - E_kin > / kBT**2 >
        gradient_u = multiply(
            _energy,
            (-1.0) / (kB * self.temperature**2),
        )
        gradient_k = (-1.0) * (3 * self.natoms - 3) / (2 * self.temperature)
        self.gradients["temperature"] = compute_sum(gradient_u, gradient_k)

    def delta_gradient(self, output: SimulationOutput):
        energies = output.get_energy(self.delta_hamiltonian)
        self.gradients["delta"] = multiply(
            take_mean(energies),
            1 / (kB * self.temperature),
        )

    def mass_gradient(output):
        raise NotImplementedError


@typeguard.typechecked
class Integration:
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        temperatures: Union[list[float], np.ndarray],
        delta_hamiltonian: Optional[Hamiltonian] = None,
        delta_coefficients: Union[list[float], np.ndarray, None] = None,
        pressure: Optional[float] = None,
    ):
        self.hamiltonian = hamiltonian
        self.temperatures = np.array(temperatures, dtype=float)
        if delta_hamiltonian is not None:
            assert delta_coefficients is not None
            self.delta_hamiltonian = delta_hamiltonian
            self.delta_coefficients = np.array(delta_coefficients, dtype=float)
        else:
            self.delta_coefficients = np.array([0.0])
            self.delta_hamiltonian = Zero()
        self.pressure = pressure

        assert len(np.unique(self.temperatures)) == len(self.temperatures)
        assert len(np.unique(self.delta_coefficients)) == len(self.delta_coefficients)

        self.states = []
        self.walkers = []
        self.outputs = []

    def create_walkers(
        self,
        dataset: Dataset,
        initialize_by: str = "quench",
        **walker_kwargs,
    ) -> list[Walker]:
        natoms = len(dataset[0].result())
        for delta in self.delta_coefficients:
            for T in self.temperatures:
                hamiltonian = self.hamiltonian + delta * self.delta_hamiltonian
                walker = Walker(
                    dataset[0],  # do quench later
                    hamiltonian,
                    temperature=T,
                    **walker_kwargs,
                )
                self.walkers.append(walker)
                state = ThermodynamicState(
                    temperature=T,
                    natoms=natoms,
                    delta_hamiltonian=self.delta_hamiltonian,
                    pressure=self.pressure,
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

    def along_delta(self, temperature: Optional[float] = None):
        if temperature is None:
            assert self.ntemperatures == 1
            temperature = self.temperatures[0]
        index = np.where(self.temperatures == temperature)[0][0]
        assert self.temperatures[index] == temperature
        N = self.ntemperatures
        states = [self.states[N * i + index] for i in range(self.ndeltas)]

        # do integration
        x = self.delta_coefficients
        y = [state.gradients["delta"] for state in states]
        f = integrate(x, *y)
        return f
        # return multiply(f, kB * temperature)

    def along_temperature(self, delta_coefficient: Optional[float] = None):
        if delta_coefficient is None:
            assert self.ndeltas == 1
            delta_coefficient = self.delta_coefficients[0]
        index = np.where(self.delta_coefficients == delta_coefficient)[0][0]
        assert self.delta_coefficients[index] == delta_coefficient
        N = self.ntemperatures
        states = [self.states[N * index + i] for i in range(self.ntemperatures)]

        # do integration
        x = self.temperatures
        y = [state.gradients["temperature"] for state in states]
        f = integrate(x, *y)
        return f
        # return multiply(f, kB * self.temperatures)

    @property
    def ntemperatures(self):
        return len(self.temperatures)

    @property
    def ndeltas(self):
        return len(self.delta_coefficients)
