from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import Union

import numpy as np
import typeguard
from ase.data import atomic_numbers
from parsl.app.app import join_app, python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, Geometry, NullState
from psiflow.utils import copy_app_future

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def _extract_energy(state: Geometry):
    if state == NullState:
        return 1e10
    else:
        return state.energy


extract_energy = python_app(_extract_energy, executors=["default_threads"])


@typeguard.typechecked
@join_app
def get_minimum_energy(element, configs, *energies):
    logger.info("atomic energies for element {}:".format(element))
    for config, energy in zip(configs, energies):
        logger.info("\t{} eV;  ".format(energy) + str(config))
    energy = min(energies)
    assert not energy == 1e10, "atomic energy calculation of {} failed".format(element)
    return copy_app_future(energy)


@typeguard.typechecked
@join_app
def evaluate_multiple(
    reference: Reference,
    nstates: int,
    inputs: list = [],
    outputs: list = [],
):
    from psiflow.data.geometry import _read_frames, write_frames

    assert len(outputs) == 1
    assert len(inputs) == 1
    states = _read_frames(inputs=[inputs[0]])
    evaluated = []
    for state in states:
        if state == NullState:
            evaluated.append(NullState)
        else:
            state = reference.evaluate_single(state)
            evaluated.append(state)
    return write_frames(
        *evaluated,
        outputs=[outputs[0]],
    )


@typeguard.typechecked
class Reference:
    properties: tuple[str, ...]

    def __init__(self, properties: tuple = ("energy", "forces")) -> None:
        self.properties = properties

    def evaluate(
        self,
        arg: Union[Dataset, Geometry, AppFuture[Geometry]],
    ) -> Union[Dataset, AppFuture]:
        if isinstance(arg, Dataset):
            data = evaluate_multiple(
                self,
                arg.length(),
                inputs=[arg.extxyz],
                outputs=[psiflow.context().new_file("data_", ".xyz")],
            )
            # to ensure the correct dependencies, it is important that
            # the output future corresponds to the actual write_dataset app.
            # otherwise, FileNotFoundErrors will occur when using HTEX.
            retval = Dataset(None, extxyz=data.outputs[0])
        else:  # Geometry, AppFuture of Geometry
            retval = self.evaluate_single(arg)
        return retval

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
            energies.append(extract_energy(reference.evaluate(state)))
        return get_minimum_energy(element, configs, *energies)

    def get_single_atom_references(self, element):
        return [(None, self)]
