from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import Union

import numpy as np
import typeguard
from ase import Atoms
from ase.data import atomic_numbers
from parsl.app.app import join_app, python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, FlowAtoms, NullState, app_write_dataset, read_dataset
from psiflow.utils import copy_app_future

logger = logging.getLogger(__name__)  # logging per module


@python_app(executors=["default_threads"])
def extract_energy(state):
    if state.reference_status:
        return state.info["energy"]
    else:
        return 1e10


@join_app
def get_minimum_energy(element, configs, *energies):
    logger.info("atomic energies for element {}:".format(element))
    for config, energy in zip(configs, energies):
        logger.info("\t{} eV;  ".format(energy) + str(config))
    energy = min(energies)
    assert not energy == 1e10, "atomic energy calculation of {} failed".format(element)
    return copy_app_future(energy)


@join_app
def evaluate_multiple(
    reference: Reference,
    nstates: int,
    inputs: list = [],
    outputs: list = [],
):
    assert len(outputs) == 1
    assert len(inputs) == 1
    data = []
    for i in range(nstates):
        state = read_dataset(i, inputs=[inputs[0]], outputs=[])
        if state == NullState:
            data.append(NullState)
        else:
            state = reference.evaluate_single(state)
            data.append(state)
    return app_write_dataset(
        None,
        return_data=True,
        inputs=data,
        outputs=[outputs[0]],
    )


@typeguard.typechecked
class Reference:
    properties: tuple[str, ...]

    def __init__(self, properties: tuple = ("energy", "forces")) -> None:
        self.properties = properties

    def evaluate(
        self,
        arg: Union[Dataset, Atoms, FlowAtoms, AppFuture],
    ) -> Union[Dataset, AppFuture]:
        if isinstance(arg, Dataset):
            data = evaluate_multiple(
                self,
                arg.length(),
                inputs=[arg.data_future],
                outputs=[psiflow.context().new_file("data_", ".xyz")],
            )
            # to ensure the correct dependencies, it is important that
            # the output future corresponds to the actual write_dataset app.
            # otherwise, FileNotFoundErrors will occur when using HTEX.
            retval = Dataset(None, data_future=data.outputs[0])
        else:  # Atoms, FlowAtoms, AppFuture
            if arg is Atoms:
                arg = FlowAtoms.from_atoms(arg)
            retval = self.evaluate_single(arg)
        return retval

    def compute_atomic_energy(self, element, box_size=None):
        energies = []
        references = self.get_single_atom_references(element)
        configs = [c for c, _ in references]
        if box_size is not None:
            atoms = FlowAtoms(
                numbers=np.array([atomic_numbers[element]]),
                positions=np.array([[0, 0, 0]]),
                cell=np.eye(3) * box_size,
                pbc=True,
            )
        else:
            atoms = FlowAtoms(
                numbers=np.array([atomic_numbers[element]]),
                positions=np.array([[0, 0, 0]]),
                pbc=False,
            )
        for _, reference in references:
            energies.append(extract_energy(reference.evaluate(atoms)))
        return get_minimum_energy(element, configs, *energies)

    def get_single_atom_references(self, element):
        return [(None, self)]
