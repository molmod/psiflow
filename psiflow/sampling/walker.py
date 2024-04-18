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
from psiflow.data import Dataset, Geometry
from psiflow.data.geometry import check_equality
from psiflow.hamiltonians.hamiltonian import Hamiltonian, Zero
from psiflow.sampling.metadynamics import Metadynamics
from psiflow.sampling.order import OrderParameter
from psiflow.utils import copy_app_future, unpack_i


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
        return deepcopy(state)


conditioned_reset = python_app(_conditioned_reset, executors=["default_threads"])


@typeguard.typechecked
@psiflow.serializable
class Walker:
    start: Union[Geometry, AppFuture]
    hamiltonian: Hamiltonian
    state: Union[Geometry, AppFuture, None]
    temperature: Optional[float]
    pressure: Optional[float]
    nbeads: int
    periodic: Optional[bool]
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
        nbeads: int = 1,
        periodic: Optional[bool] = None,
        timestep: float = 0.5,
        metadynamics: Optional[Metadynamics] = None,
        order_parameter: Optional[OrderParameter] = None,
    ):
        if type(start) is AppFuture:
            start = start.result()  # blocking
        if periodic is not None:
            assert periodic == bool(start.periodic)
        else:
            periodic = bool(start.periodic)
        self.start = start
        self.periodic = periodic
        if hamiltonian is None:
            hamiltonian = Zero()
        self.hamiltonian = hamiltonian
        if state is None:
            state = copy_app_future(self.start)
        self.state = state

        if order_parameter is not None:
            self.start = order_parameter.evaluate(self.start)

        self.temperature = temperature
        if pressure is not None:
            assert self.periodic
        self.pressure = pressure
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
                nbeads=self.nbeads,
                periodic=self.periodic,
                timestep=self.timestep,
                metadynamics=metadynamics,
            )  # no coupling
            walkers.append(walker)
        return walkers

    @staticmethod
    def is_similar(w0: Walker, w1: Walker):
        similar_T = (w0.temperature is None) == (w1.temperature is None)
        similar_P = (w0.pressure is None) == (w1.pressure is None)
        similar_pimd = w0.nbeads == w1.nbeads
        similar_coupling = w0.coupling == w1.coupling
        similar_periodic = w0.periodic == w1.periodic
        similar_timestep = w0.timestep == w1.timestep
        return (
            similar_T
            and similar_P
            and similar_pimd
            and similar_coupling
            and similar_periodic
            and similar_timestep
        )

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
def partition(walkers: list[Walker]) -> list[list[Walker]]:
    partitions = []
    for walker in walkers:
        found = False
        for partition in partitions:
            if Walker.is_similar(walker, partition[0]):
                partition.append(walker)
                found = True
        if not found:
            partitions.append([walker])
    return partitions


@typeguard.typechecked
def _get_minimum_energy_states(
    coefficients: np.ndarray,
    inputs: list = [],
) -> tuple[int, ...]:
    import numpy

    from psiflow.data.geometry import _read_frames

    assert len(coefficients.shape) == 2
    assert len(inputs) == coefficients.shape[1]

    energies = []
    for i in range(len(inputs)):
        data = _read_frames(inputs=[inputs[i]])
        energies.append([g.energy for g in data])
    energies = numpy.array(energies)

    indices = []
    for c in coefficients:
        energy = numpy.sum(c.reshape(-1, 1) * energies, axis=0)
        indices.append(int(numpy.argmin(energy)))
    return tuple(indices)


get_minimum_energy_states = python_app(
    _get_minimum_energy_states,
    executors=["default_threads"],
)


@typeguard.typechecked
def quench(walkers: list[Walker], dataset: Dataset) -> None:
    all_hamiltonians = sum([w.hamiltonian for w in walkers], start=Zero())
    evaluated = [h.evaluate(dataset) for h in all_hamiltonians.hamiltonians]

    coefficients = []
    for walker in walkers:
        c = all_hamiltonians.get_coefficients(1.0 * walker.hamiltonian)
        coefficients.append(c)
    coefficients = np.array(coefficients)

    indices = get_minimum_energy_states(
        coefficients,
        inputs=[data.extxyz for data in evaluated],
    )
    for i, walker in enumerate(walkers):
        walker.start = dataset[unpack_i(indices, i)][0]
        walker.reset()


def _random_indices(nindices: int, nstates: int) -> list[int]:
    indices = np.random.randint(0, high=nstates, size=(nindices,))
    return [int(i) for i in indices]


random_indices = python_app(_random_indices, executors=["default_threads"])


@typeguard.typechecked
def randomize(walkers: list[Walker], dataset: Dataset) -> None:
    indices = random_indices(len(walkers), dataset.length())
    for i, walker in enumerate(walkers):
        walker.start = dataset[unpack_i(indices, i)][0]
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
    assert len(partition(walkers)) == 1
    assert walkers[0].coupling is None
    rex = ReplicaExchange(trial_frequency, rescale_kinetic, len(walkers))
    for walker in walkers:
        walker.coupling = rex
