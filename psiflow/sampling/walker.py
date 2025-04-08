from __future__ import annotations  # necessary for type-guarding class methods

import xml.etree.ElementTree as ET
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry, check_equality
from psiflow.hamiltonians import Hamiltonian, Zero
from psiflow.order_parameters import OrderParameter
from psiflow.sampling.metadynamics import Metadynamics
from psiflow.utils.apps import copy_app_future


@typeguard.typechecked
class Coupling:
    pass


@typeguard.typechecked
def _conditioned_reset(
    condition: bool,
    state: Geometry,
    start: Geometry,
) -> Geometry:
    from copy import deepcopy  # copy necessary!

    if condition:
        return deepcopy(start)
    else:
        return state


conditioned_reset = python_app(_conditioned_reset, executors=["default_threads"])


@typeguard.typechecked
@psiflow.serializable
class Walker:
    start: Union[Geometry, AppFuture]
    hamiltonian: Optional[Hamiltonian]
    state: Union[Geometry, AppFuture, None]
    temperature: Optional[float]
    pressure: Optional[float]
    masses: Union[np.ndarray, float, None]
    nbeads: int
    timestep: float
    coupling: Optional[Coupling]
    metadynamics: Optional[Metadynamics]
    order_parameter: Optional[OrderParameter]

    def __init__(
        self,
        start: Union[Geometry, AppFuture],
        hamiltonian: Optional[Hamiltonian] = None,
        state: Union[Geometry, AppFuture, None] = None,
        temperature: Optional[float] = 300,
        pressure: Optional[float] = None,
        masses: Union[np.ndarray, float, None] = None,
        nbeads: int = 1,
        timestep: float = 0.5,
        metadynamics: Optional[Metadynamics] = None,
        order_parameter: Optional[OrderParameter] = None,
    ):
        self.start = start
        if hamiltonian is None:
            hamiltonian = Zero()
        self.hamiltonian = hamiltonian
        if state is None:
            state = copy_app_future(self.start)
        self.state = state

        if order_parameter is not None:
            self.start = order_parameter.evaluate(self.start)

        self.temperature = temperature
        self.pressure = pressure

        if isinstance(masses, (float, int)):
            masses *= self.start.atomic_masses
        self.masses = masses
        if self.masses is not None:
            assert len(self.masses) == len(self.start), "Masses do not match number of atoms"

        self.nbeads = nbeads
        self.timestep = timestep

        self.metadynamics = metadynamics
        self.order_parameter = order_parameter
        self.coupling = None

    def reset(self, condition: Union[AppFuture, bool] = True):
        self.state = conditioned_reset(condition, self.state, self.start)

    def is_reset(self) -> AppFuture:
        return check_equality(self.start, self.state)

    def multiply(self, nreplicas: int) -> list[Walker]:
        if self.coupling is not None:
            raise ValueError("Cannot multiply walkers after they are coupled")
        walkers = []
        for _i in range(nreplicas):
            if self.metadynamics is not None:
                metadynamics = self.metadynamics.copy()
            else:
                metadynamics = None
            walker = Walker(
                start=self.start,
                hamiltonian=self.hamiltonian,
                state=self.state,
                temperature=self.temperature,
                pressure=self.pressure,
                masses=self.masses,
                nbeads=self.nbeads,
                timestep=self.timestep,
                metadynamics=metadynamics,
            )  # no coupling
            walkers.append(walker)
        return walkers

    @property
    def pimd(self):
        return self.nbeads != 1

    @property
    def nve(self):
        return (self.temperature is None) and (self.pressure is None)

    @property
    def nvt(self):
        return (self.temperature is not None) and (self.pressure is None)

    @property
    def npt(self):
        return (self.temperature is not None) and (self.pressure is not None)


@typeguard.typechecked
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


# typeguarding incompatible with * expansion
def _get_minimum_energy_states(
    coefficients: np.ndarray,
    *energies: np.ndarray,
) -> list[int]:
    import numpy

    assert len(coefficients.shape) == 2
    assert len(energies) == coefficients.shape[1]

    energies = numpy.array(energies)

    indices = []
    for c in coefficients:
        energy = numpy.sum(c.reshape(-1, 1) * energies, axis=0)
        indices.append(int(numpy.argmin(energy)))
    return indices


get_minimum_energy_states = python_app(
    _get_minimum_energy_states,
    executors=["default_threads"],
)


@typeguard.typechecked
def quench(walkers: list[Walker], dataset: Dataset) -> None:
    all_hamiltonians = sum([w.hamiltonian for w in walkers], start=Zero())
    energies = [h.compute(dataset, "energy") for h in all_hamiltonians.hamiltonians]

    coefficients = []
    for walker in walkers:
        c = all_hamiltonians.get_coefficients(1.0 * walker.hamiltonian)
        assert c is not None
        coefficients.append(c)
    coefficients = np.array(coefficients)

    indices = get_minimum_energy_states(
        coefficients,
        *energies,
    )
    data = dataset[indices]
    for i, walker in enumerate(walkers):
        walker.start = data[i]
        walker.reset()


def _random_indices(nindices: int, nstates: int) -> list[int]:
    indices = np.random.randint(0, high=nstates, size=(nindices,))
    return [int(i) for i in indices]


random_indices = python_app(_random_indices, executors=["default_threads"])


@typeguard.typechecked
def randomize(walkers: list[Walker], dataset: Dataset) -> None:
    indices = random_indices(len(walkers), dataset.length())
    data = dataset[indices]
    for i, walker in enumerate(walkers):
        walker.start = data[i]
        walker.reset()


@typeguard.typechecked
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


@typeguard.typechecked
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

    def __eq__(self, other: Optional[ReplicaExchange]) -> bool:
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
