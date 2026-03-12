import xml.etree.ElementTree as ET
from typing import Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry, check_equality
from psiflow.hamiltonians import Hamiltonian, Zero, combine_hamiltonians
from psiflow.order_parameters import OrderParameter
from psiflow.sampling.metadynamics import Metadynamics
from psiflow.utils.apps import copy_app_future


class Coupling:
    pass


class Ensemble(Enum):
    NVE = "NVE"
    NVT = "NVT"
    NPT = "NPT"
    NVST = "(N, V, sigma=0, T)"


def _conditional_reset(
    state: Geometry,
    start: Geometry,
    flag: bool | None,
    condition: Callable | None,
    *args,
) -> Geometry:
    """Reset geometry based on a flag or boolean condition function."""
    if condition is not None:
        flag = condition(args)

    # copy necessary!
    return deepcopy(start) if flag else state


conditional_reset = python_app(_conditional_reset, executors=["default_threads"])


def get_ensemble_kwargs(walker: "Walker") -> dict:
    """Extract all walker properties that define the MD sampling ensemble."""
    return dict(
        hamiltonian=walker.hamiltonian,
        timestep=walker.timestep,
        temperature=walker.temperature,
        pressure=walker.pressure,
        tau_thermostat=walker.tau_thermostat,
        tau_barostat=walker.tau_barostat,
        volume_constrained=walker.volume_constrained,
        masses=walker.masses,
        nbeads=walker.nbeads,
        metadynamics=walker.metadynamics,
    )


@psiflow.serializable
@dataclass
class Walker:
    start: Union[Geometry, AppFuture]
    hamiltonian: Hamiltonian = Zero()
    timestep: float = 0.5
    temperature: Optional[float] = 300
    pressure: Optional[float] = None
    tau_thermostat: float = 100
    tau_barostat: float = 300
    volume_constrained: bool = False
    masses: Union[np.ndarray, float, None] = None
    nbeads: int = 1
    metadynamics: Optional[Metadynamics] = None
    order_parameter: Optional[OrderParameter] = None

    state: Union[Geometry, AppFuture] = field(init=False)
    coupling: Optional[Coupling] = field(init=False)

    def __post_init__(self):
        self.state = copy_app_future(self.start)
        self.coupling = None

        if self.temperature is None:  # NVE
            assert self.pressure is None and not self.volume_constrained
        if self.volume_constrained:
            self.pressure = 0  # TODO: warning?

        if isinstance(self.start, Geometry) and not self.start.periodic:
            # we cannot check this for futures
            assert self.pressure is None, "Pressure requires PBC"

        if self.order_parameter is not None:
            # TODO: order_parameter out of commission
            self.start = self.order_parameter.evaluate(self.start)

        if (m := self.masses) is None:
            pass  # do nothing
        elif isinstance(m, (float, int)):
            self.masses = self.start.atomic_masses * m
        elif isinstance(m, np.ndarray) and len(m) != len(self.start):
            raise ValueError("Supplied masses do not match number of atoms")

    def reset(self):
        self.state = conditional_reset(self.state, self.start, True, None)

    def conditional_reset(
        self, flag: bool | None = None, condition: Callable | None = None, *args
    ):
        assert (flag is None) != (condition is None)  # xor
        self.state = conditional_reset(self.state, self.start, flag, condition, *args)

    def is_reset(self) -> AppFuture:
        return check_equality(self.start, self.state)

    def multiply(self, nreplicas: int) -> list["Walker"]:
        if self.coupling is not None:
            raise ValueError("Cannot multiply walkers after they are coupled")
        walkers = []
        kwargs = get_ensemble_kwargs(self)
        for _i in range(nreplicas):
            if kwargs["metadynamics"] is not None:
                kwargs["metadynamics"] = kwargs["metadynamics"].copy()
            walker = Walker(start=self.start, **kwargs)  # no coupling
            walker.state = self.state
            walkers.append(walker)
        return walkers

    @property
    def pimd(self) -> bool:
        return self.nbeads != 1

    @property
    def ensemble(self) -> Ensemble:
        if self.temperature is None:
            return Ensemble.NVE
        elif self.pressure is None:
            return Ensemble.NVT
        elif self.volume_constrained:
            return Ensemble.NVST
        else:
            return Ensemble.NPT


def partition(walkers: list[Walker]) -> list[list[int]]:
    indices = []
    for i, walker in enumerate(walkers):
        found = False
        if walker.coupling is not None:
            for group in indices:
                if walker.coupling == walkers[group[0]].coupling:
                    assert not found
                    group.append(i)
                    found = True
        if not found:
            indices.append([i])
    return indices


def _get_minimum_energy_states(
    coefficients: np.ndarray, *energies: np.ndarray
) -> list[int]:
    assert len(coefficients.shape) == 2
    assert len(energies) == coefficients.shape[1]

    indices = []
    energies = np.array(energies)
    for c in coefficients:  # every c corresponds to a Walker hamiltonian
        energy = np.sum(c.reshape(-1, 1) * energies, axis=0)
        indices.append(int(np.argmin(energy)))
    return indices


get_minimum_energy_states = python_app(
    _get_minimum_energy_states, executors=["default_threads"]
)


def quench(walkers: list[Walker], dataset: Dataset) -> None:
    """Assign the lowest energy geometry in dataset to every walker"""
    hamiltonians = combine_hamiltonians([w.hamiltonian for w in walkers])
    energies = [h.compute(dataset, "energy") for h in hamiltonians.hamiltonians]
    coefficients = [
        hamiltonians.get_coefficients(walker.hamiltonian * 1.0) for walker in walkers
    ]
    indices = get_minimum_energy_states(np.array(coefficients), *energies)
    geometries = dataset.geometries()
    for walker, idx in zip(walkers, indices):
        walker.start = geometries[idx]
        walker.reset()


def _random_indices(nindices: int, nstates: int) -> list[int]:
    indices = np.random.randint(0, high=nstates, size=(nindices,))
    return [int(i) for i in indices]


random_indices = python_app(_random_indices, executors=["default_threads"])


def randomize(walkers: list[Walker], dataset: Dataset) -> None:
    """Randomly assign geometries from dataset to walkers"""
    indices = random_indices(len(walkers), dataset.length())
    geometries = dataset.geometries()
    for walker, idx in zip(walkers, indices):
        walker.start = geometries[idx]
        walker.reset()


def validate_coupling(walkers: list[Walker]):
    couplings = []
    counts = []
    for walker in walkers:
        coupling = walker.coupling
        if coupling is None:
            continue
        if coupling not in couplings:
            couplings.append(couplings)
            counts.append(1)
        else:
            index = couplings.index(coupling)
            counts[index] += 1
    for i, coupling in enumerate(couplings):
        assert coupling.nwalkers == counts[i]


@psiflow.serializable
class ReplicaExchange(Coupling):
    trial_frequency: int
    rescale_kinetic: bool
    nwalkers: int
    swapfile: psiflow._DataFuture

    def __init__(
        self,
        trial_frequency: int,
        rescale_kinetic: bool,
        nwalkers: int,  # purely for safety!
    ) -> None:
        self.trial_frequency = trial_frequency
        self.rescale_kinetic = rescale_kinetic
        self.swapfile = psiflow.context().new_file("swap_", ".txt")
        self.nwalkers = nwalkers

    def __eq__(self, other: Optional["ReplicaExchange"]) -> bool:
        if other is None:
            return False
        trial = self.trial_frequency == other.trial_frequency
        rescale = self.rescale_kinetic == other.rescale_kinetic
        swapfile = self.swapfile.filepath == other.swapfile.filepath
        return trial and rescale and swapfile

    def inputs(self) -> list[Union[DataFuture, File]]:
        return [self.swapfile]

    def get_smotion(self, has_metad: bool) -> ET.Element:
        remd = ET.Element("remd")
        stride = ET.Element("stride")
        stride.text = str(self.trial_frequency)
        remd.append(stride)
        krescale = ET.Element("krescale")
        krescale.text = str(self.rescale_kinetic)
        remd.append(krescale)
        swapfile = ET.Element("swapfile")
        swapfile.text = "replica_exchange"  # gets prefixed by i-PI
        remd.append(swapfile)
        if has_metad:
            smotion = ET.Element("smotion", mode="multi")
            smotion_remd = ET.Element("smotion", mode="remd")
            smotion_remd.append(remd)
            smotion.append(smotion_remd)
        else:
            smotion = ET.Element("smotion", mode="remd")
            smotion.append(remd)
        return smotion

    def update(self, result: AppFuture):
        self.swapfile = result.outputs[-1]

    def copy_command(self):
        return "cp output.replica_exchange {}".format(self.swapfile.filepath)


def replica_exchange(
    walkers: list[Walker],
    trial_frequency: int = 50,
    rescale_kinetic: bool = True,
) -> None:
    for w in walkers:
        assert w.coupling is None
    rex = ReplicaExchange(trial_frequency, rescale_kinetic, len(walkers))
    for walker in walkers:
        walker.coupling = rex
