from __future__ import annotations  # necessary for type-guarding class methods

import logging
from copy import deepcopy
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


@typeguard.typechecked
class BaseReference:
    def __init__(self, properties: tuple = ("energy", "forces")) -> None:
        self.properties = properties
        try:
            self.__class__.create_apps()
        except AssertionError:
            pass  # apps already created

    def evaluate(
        self,
        arg: Union[Dataset, Atoms, FlowAtoms, AppFuture],
    ) -> Union[Dataset, AppFuture]:
        context = psiflow.context()
        if isinstance(arg, Dataset):
            data = context.apps(self.__class__, "evaluate_multiple")(
                deepcopy(self.parameters),
                arg.length(),
                inputs=[arg.data_future],
                outputs=[context.new_file("data_", ".xyz")],
            )
            # to ensure the correct dependencies, it is important that
            # the output future corresponds to the actual write_dataset app.
            # otherwise, FileNotFoundErrors will occur when using HTEX.
            retval = Dataset(None, data_future=data.outputs[0])
        else:  # Atoms, FlowAtoms, AppFuture
            if arg is Atoms:
                arg = FlowAtoms.from_atoms(arg)
            data = context.apps(self.__class__, "evaluate_single")(
                arg,  # converts to FlowAtoms if necessary
                deepcopy(self.parameters),
            )
            retval = data
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

    @property
    def parameters(self):
        raise NotImplementedError

    @classmethod
    def create_apps(cls) -> None:
        assert not (cls == BaseReference)  # should never be called directly
        context = psiflow.context()

        def evaluate_multiple(
            parameters,
            nstates,
            inputs=[],
            outputs=[],
        ):
            assert len(outputs) == 1
            assert len(inputs) == 1
            data = []
            for i in range(nstates):
                state = read_dataset(i, inputs=[inputs[0]], outputs=[])
                if state == NullState:
                    data.append(NullState)
                else:
                    data.append(
                        context.apps(cls, "evaluate_single")(
                            state,
                            parameters,
                        )
                    )
            return app_write_dataset(
                None,
                return_data=True,
                inputs=data,
                outputs=[outputs[0]],
            )

        app_evaluate_multiple = join_app(evaluate_multiple)
        context.register_app(cls, "evaluate_multiple", app_evaluate_multiple)
