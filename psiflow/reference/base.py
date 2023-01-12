from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Tuple
import typeguard
from dataclasses import dataclass
from copy import deepcopy

from parsl.dataflow.futures import AppFuture
from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File

from ase import Atoms

from psiflow.execution import Container, ExecutionContext
from psiflow.data import FlowAtoms, Dataset, read_dataset, save_dataset, \
        get_length_dataset
from psiflow.utils import copy_app_future, unpack_i, combine_futures


@typeguard.typechecked
def _separate_failed_states(
        data: List[FlowAtoms],
        ) -> Tuple[List[FlowAtoms], List[str]]:
    logs   = []
    failed = []
    for atoms in data:
        if not atoms.reference_status:
            assert atoms.reference_log is not None
            logs.append(atoms.reference_log)
            failed.append(atoms)
    return failed, logs
separate_failed_states = python_app(_separate_failed_states, executors=['default'])


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

    def __init__(self, context: ExecutionContext, **kwargs) -> None:
        super().__init__(context)
        self.parameters  = self.parameters_cls(**deepcopy(kwargs))
        self.data_failed = Dataset(context, []) # initialize empty
        self.logs        = copy_app_future([])

    def evaluate(
            self,
            arg: Union[Dataset, Atoms, FlowAtoms, AppFuture],
            ) -> Union[Dataset, AppFuture]:
        parameters = deepcopy(self.parameters)
        if isinstance(arg, Dataset):
            data = self.context.apps(self.__class__, 'evaluate_multiple')(
                    parameters,
                    arg.length(),
                    inputs=[arg.data_future],
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
                    )
            retval = data
            data = combine_futures(inputs=[data])
        failed_logs = separate_failed_states(data)
        failed = unpack_i(failed_logs, 0)
        logs   = unpack_i(failed_logs, 1)
        self.data_failed.append(Dataset(self.context, failed))
        self.logs = update_logs(self.logs, logs)
        return retval

    @classmethod
    def create_apps(cls, context: ExecutionContext) -> None:
        assert not (cls == BaseReference) # should never be called directly
        def evaluate_multiple(parameters, nstates, inputs=[], outputs=[]):
            assert len(outputs) == 1
            data = []
            for i in range(nstates):
                data.append(context.apps(cls, 'evaluate_single')(
                    read_dataset(i, inputs=[inputs[0]], outputs=[]),
                    parameters,
                    inputs=[],
                    outputs=[],
                    ))
            return context.apps(Dataset, 'save_dataset')(
                    None,
                    return_data=True,
                    inputs=data,
                    outputs=[outputs[0]],
                    )
            #return combine_futures(inputs=data) # join apps only return single future
        app_evaluate_multiple = join_app(evaluate_multiple)
        context.register_app(cls, 'evaluate_multiple', app_evaluate_multiple)
