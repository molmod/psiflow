from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Tuple
import typeguard
from dataclasses import dataclass
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

from parsl.dataflow.futures import AppFuture
from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File

from ase import Atoms

from psiflow.execution import Container, ExecutionContext
from psiflow.data import FlowAtoms, Dataset, read_dataset, app_save_dataset, \
        get_length_dataset
from psiflow.utils import copy_app_future, unpack_i, combine_futures


@typeguard.typechecked
def _update_logs(logs: List[str], new_logs: List[str]) -> List[str]:
    return logs + new_logs
update_logs = python_app(_update_logs, executors=['default'])


@typeguard.typechecked
@dataclass
class EmptyParameters:
    pass


@typeguard.typechecked
class BaseReference(Container):
    parameters_cls = EmptyParameters
    required_files = []

    def __init__(self, context: ExecutionContext, **kwargs) -> None:
        super().__init__(context)
        self.parameters = self.parameters_cls(**deepcopy(kwargs))
        self.files    = {}
        try:
            self.__class__.create_apps(context)
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
        parameters = deepcopy(self.parameters)
        if isinstance(arg, Dataset):
            data = self.context.apps(self.__class__, 'evaluate_multiple')(
                    parameters,
                    arg.length(),
                    file_names=list(self.files.keys()),
                    inputs=[arg.data_future] + list(self.files.values()),
                    outputs=[self.context.new_file('data_', '.xyz')],
                    )
            # to ensure the correct dependencies, it is important that
            # the output future corresponds to the actual save_dataset app.
            # otherwise, FileNotFoundErrors will occur when using HTEX.
            retval = Dataset(self.context, None, data_future=data.outputs[0])
        else: # Atoms, FlowAtoms, AppFuture
            data = self.context.apps(self.__class__, 'evaluate_single')(
                    arg, # converts to FlowAtoms if necessary
                    parameters,
                    file_names=list(self.files.keys()),
                    inputs=list(self.files.values()),
                    )
            retval = data
            #data = combine_futures(inputs=[data])
        #failed_logs = separate_failed_states(data)
        #failed = unpack_i(failed_logs, 0)
        #logs   = unpack_i(failed_logs, 1)
        #self.states_failed.append(Dataset(self.context, failed))
        #self.logs = update_logs(self.logs, logs)
        return retval

    @classmethod
    def create_apps(cls, context: ExecutionContext) -> None:
        assert not (cls == BaseReference) # should never be called directly
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
