from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Tuple, ClassVar
import typeguard
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

from parsl.dataflow.futures import AppFuture
from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File

from ase import Atoms

import psiflow
from psiflow.data import FlowAtoms, Dataset, read_dataset, app_save_dataset, \
        get_length_dataset
from psiflow.utils import copy_app_future, unpack_i, combine_futures
#from .utils import generate_isolated_atoms


@python_app(executors=['default'])
def extract_energies(inputs=[]):
    from psiflow.data import read_dataset
    data = read_dataset

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

    def get_atomic_energies(self, elements, box_size=5):
        dataset = isolated_atoms_dataset(elements, box_size)
        evaluated = self.evaluate(dataset)
        return extract_energies

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
