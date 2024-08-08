from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import ClassVar, Optional, Union

import numpy as np
import parsl
import typeguard
from ase.data import atomic_numbers
from parsl.app.app import join_app, python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Computable, Dataset
from psiflow.geometry import Geometry, NullState
from psiflow.utils.apps import copy_app_future, unpack_i

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def _extract_energy(state: Geometry):
    if state == NullState:
        return 1e10
    else:
        return state.energy


extract_energy = python_app(_extract_energy, executors=["default_threads"])


@join_app
@typeguard.typechecked
def get_minimum_energy(element, configs, *energies):
    logger.info("atomic energies for element {}:".format(element))
    for config, energy in zip(configs, energies):
        logger.info("\t{} eV;  ".format(energy) + str(config))
    energy = min(energies)
    assert not energy == 1e10, "atomic energy calculation of {} failed".format(element)
    return copy_app_future(energy)


@join_app
@typeguard.typechecked
def evaluate(
    geometry: Geometry,
    reference: Reference,
) -> AppFuture:
    if geometry == NullState:
        return copy_app_future(NullState)
    else:
        future = reference.app_pre(
            geometry,
            stdout=parsl.AUTO_LOGNAME,
            stderr=parsl.AUTO_LOGNAME,
        )
        return reference.app_post(
            geometry=geometry.copy(),
            inputs=[future.stdout, future.stderr, future],
        )


@join_app
@typeguard.typechecked
def compute_dataset(
    dataset: Dataset,
    length: int,
    reference: Reference,
) -> AppFuture:
    from psiflow.data.utils import extract_quantities

    geometries = dataset.geometries()  # read it once
    evaluated = [evaluate(unpack_i(geometries, i), reference) for i in range(length)]
    future = extract_quantities(
        tuple(reference.outputs),
        None,
        None,
        *evaluated,
    )
    return future


@typeguard.typechecked
@psiflow.serializable
class Reference(Computable):
    outputs: tuple
    batch_size: ClassVar[int] = 1  # not really used

    def compute(self, dataset: Dataset, *outputs: Optional[Union[str, tuple]]):
        compute_outputs = compute_dataset(dataset, dataset.length(), self)
        if len(outputs) == 0:
            outputs_ = tuple(self.outputs)
        else:
            outputs_ = outputs
        to_return = []
        for output in outputs_:
            if output not in self.outputs:
                raise ValueError("output {} not in {}".format(output, self.outputs))
            index = outputs_.index(output)
            to_return.append(compute_outputs[index])
        if len(outputs_) == 1:
            return to_return[0]
        else:
            return tuple(to_return)

    def compute_atomic_energy(self, element, box_size=None):
        energies = []
        references = self.get_single_atom_references(element)
        configs = [c for c, _ in references]
        if box_size is not None:
            state = Geometry.from_data(
                numbers=np.array([atomic_numbers[element]]),
                positions=np.array([[0, 0, 0]]),
                cell=np.eye(3) * box_size,
            )
        else:
            state = Geometry(
                numbers=np.array([atomic_numbers[element]]),
                positions=np.array([[0, 0, 0]]),
                cell=np.zeros((3, 3)),
            )
        for _, reference in references:
            energies.append(extract_energy(evaluate(state, reference)))
        return get_minimum_energy(element, configs, *energies)

    def get_single_atom_references(self, element):
        return [(None, self)]
