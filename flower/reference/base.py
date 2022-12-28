from dataclasses import dataclass
from copy import deepcopy

from parsl.dataflow.futures import AppFuture
from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File

from ase import Atoms

from flower.execution import Container
from flower.data import FlowerAtoms, Dataset, read_dataset, save_dataset, \
        get_length_dataset
from flower.utils import _new_file


@dataclass
class EmptyParameters:
    pass


class BaseReference(Container):
    parameters_cls = EmptyParameters

    def __init__(self, context, **kwargs):
        super().__init__(context)
        self.parameters = self.parameters_cls(**deepcopy(kwargs))

    def evaluate(self, arg):
        parameters = deepcopy(self.parameters)
        if isinstance(arg, Dataset):
            data_future = self.context.apps(self.__class__, 'evaluate_multiple')(
                    parameters,
                    inputs=[arg.data_future],
                    outputs=[File(_new_file(self.context.path, 'data_', '.xyz'))],
                    ).outputs[0]
            return Dataset(self.context, data_future=data_future)
        else:
            if type(arg) == Atoms: # convert to FlowerAtoms
                arg = FlowerAtoms.from_atoms(arg)
            assert (isinstance(arg, FlowerAtoms) or isinstance(arg, AppFuture))
            return self.context.apps(self.__class__, 'evaluate_single')(
                    arg,
                    parameters,
                    inputs=[],
                    outputs=[],
                    )

    @classmethod
    def create_apps(cls, context):
        assert not (cls == BaseReference) # should never be called directly
        app_length_dataset = context.apps(Dataset, 'length_dataset')
        app_save_dataset   = context.apps(Dataset, 'save_dataset')

        def evaluate_multiple(parameters, inputs=[], outputs=[]):
            assert len(outputs) == 1
            data = []
            nstates = app_length_dataset(inputs=[inputs[0]])
            for i in range(nstates.result()):
                data.append(context.apps(cls, 'evaluate_single')(
                    read_dataset(i, inputs=[inputs[0]], outputs=[]),
                    parameters,
                    inputs=[],
                    outputs=[],
                    ))
            for i in range(nstates.result()):
                print(data[i].result().info['evaluation_flag'])
            return app_save_dataset(states=None, inputs=data, outputs=[outputs[0]])
        app_evaluate_multiple = join_app(evaluate_multiple)
        context.register_app(cls, 'evaluate_multiple', app_evaluate_multiple)
