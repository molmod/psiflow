from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List
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


@typeguard.typechecked
@dataclass
class EmptyParameters:
    pass


@typeguard.typechecked
class BaseReference(Container):
    parameters_cls = EmptyParameters

    def __init__(self, context: ExecutionContext, **kwargs) -> None:
        super().__init__(context)
        self.parameters = self.parameters_cls(**deepcopy(kwargs))

    def evaluate(
            self,
            arg: Union[Dataset, Atoms, FlowAtoms, AppFuture],
            ) -> Union[Dataset, AppFuture]:
        parameters = deepcopy(self.parameters)
        if isinstance(arg, Dataset):
            data_future = self.context.apps(self.__class__, 'evaluate_multiple')(
                    parameters,
                    arg.length(),
                    inputs=[arg.data_future],
                    outputs=[self.context.new_file('data_', '.xyz')],
                    ).outputs[0]
            return Dataset(self.context, None, data_future=data_future)
        else:
            if type(arg) == Atoms: # convert to FlowAtoms
                arg = FlowAtoms.from_atoms(arg)
            assert (isinstance(arg, FlowAtoms) or isinstance(arg, AppFuture))
            return self.context.apps(self.__class__, 'evaluate_single')(
                    arg,
                    parameters,
                    inputs=[],
                    outputs=[],
                    )

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
                    states=None,
                    inputs=data,
                    outputs=[outputs[0]],
                    )
        app_evaluate_multiple = join_app(evaluate_multiple)
        context.register_app(cls, 'evaluate_multiple', app_evaluate_multiple)
