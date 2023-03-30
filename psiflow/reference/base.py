from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Tuple, ClassVar
import typeguard
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import numpy as np
import logging

from parsl.dataflow.futures import AppFuture
from parsl.app.app import join_app, python_app, join_app
from parsl.data_provider.files import File

from ase import Atoms
from ase.data import atomic_numbers

import psiflow
from psiflow.data import FlowAtoms, Dataset, read_dataset, app_save_dataset, \
        get_length_dataset
from psiflow.utils import copy_app_future, unpack_i, combine_futures


logger = logging.getLogger(__name__) # logging per module


@python_app(executors=['default'])
def extract_energy(state):
    if state.reference_status:
        return state.info['energy']
    else:
        return 1e10


@join_app
def get_minimum_energy(element, configs, *energies):
    index = energies.index(min(energies))
    logger.info('atomic energies for element {}:'.format(element))
    for config, energy in zip(configs, energies):
        logger.info('\t{} eV;  '.format(energy) + str(config))
    energy = min(energies)
    assert not energy == 0.0, 'atomic energy calculations failed'
    return copy_app_future(energy)


@typeguard.typechecked
class BaseReference:
    required_files = []

    def __init__(self) -> None:
        self.files = {}
        try:
            self.__class__.create_apps()
        except AssertionError:
            pass # apps already created

    def add_file(self, name: str, file: Union[Path, str, File]):
        assert name in self.required_files
        if not isinstance(file, File):
            file = File(str(file))
        self.files[name] = file

    def evaluate(
            self,
            arg: Union[Dataset, Atoms, FlowAtoms, AppFuture],
            ) -> Union[Dataset, AppFuture]:
        for name in self.required_files:
            assert name in self.files.keys()
            assert Path(self.files[name].filepath).is_file()
        context = psiflow.context()
        if isinstance(arg, Dataset):
            data = context.apps(self.__class__, 'evaluate_multiple')(
                    deepcopy(self.parameters),
                    arg.length(),
                    file_names=list(self.files.keys()),
                    inputs=[arg.data_future] + list(self.files.values()),
                    outputs=[context.new_file('data_', '.xyz')],
                    )
            # to ensure the correct dependencies, it is important that
            # the output future corresponds to the actual save_dataset app.
            # otherwise, FileNotFoundErrors will occur when using HTEX.
            retval = Dataset(None, data_future=data.outputs[0])
        else: # Atoms, FlowAtoms, AppFuture
            data = context.apps(self.__class__, 'evaluate_single')(
                    arg, # converts to FlowAtoms if necessary
                    deepcopy(self.parameters),
                    file_names=list(self.files.keys()),
                    inputs=list(self.files.values()),
                    )
            retval = data
        return retval

    def compute_atomic_energy(self, element, box_size=5):
        energies = []
        references = self.get_single_atom_references(element)
        configs  = [c for c, _ in references]
        atoms = FlowAtoms(
                numbers=np.array([atomic_numbers[element]]),
                positions=np.array([[0, 0, 0]]),
                cell=np.eye(3) * box_size,
                pbc=True,
                )
        for config, reference in references:
            energies.append(extract_energy(reference.evaluate(atoms)))
        return get_minimum_energy(element, configs, *energies)

    def get_single_atom_references(self, element):
        return [(None, self)]

    @property
    def parameters(self):
        raise NotImplementedError

    @classmethod
    def create_apps(cls) -> None:
        assert not (cls == BaseReference) # should never be called directly
        context = psiflow.context()
        def evaluate_multiple(
                parameters,
                nstates,
                file_names,
                inputs=[],
                outputs=[],
                ):
            assert len(outputs) == 1
            assert len(inputs) == len(cls.required_files) + 1
            data = []
            for i in range(nstates):
                data.append(context.apps(cls, 'evaluate_single')(
                    read_dataset(i, inputs=[inputs[0]], outputs=[]),
                    parameters,
                    file_names,
                    inputs=inputs[1:],
                    ))
            return app_save_dataset(
                    None,
                    return_data=True,
                    inputs=data,
                    outputs=[outputs[0]],
                    )
        app_evaluate_multiple = join_app(evaluate_multiple)
        context.register_app(cls, 'evaluate_multiple', app_evaluate_multiple)