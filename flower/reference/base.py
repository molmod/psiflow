from dataclasses import dataclass

from parsl.dataflow.futures import AppFuture
from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File

from ase import Atoms

from flower.execution import Container, ReferenceExecutionDefinition
from flower.dataset import Dataset, _new_xyz, read_dataset, save_dataset, \
        get_length_dataset
from flower.utils import _new_file


@dataclass
class EmptyParameters:
    pass


class BaseReference(Container):
    parameters_cls = EmptyParameters

    def __init__(self, context, **kwargs):
        super().__init__(context)
        self.parameters = self.parameters_cls(**kwargs)

    def evaluate(self, arg):
        if isinstance(arg, Dataset):
            data_future = self.context.apps(self.__class__, 'evaluate_multiple')(
                    self.parameters,
                    inputs=[arg.data_future],
                    outputs=[File(_new_xyz(self.context))],
                    ).outputs[0]
            return Dataset(self.context, data_future=data_future)
        else:
            # should be either atoms or AppFuture of atoms instance
            assert (isinstance(arg, Atoms) or isinstance(arg, AppFuture))
            return self.context.apps(self.__class__, 'evaluate_single')(
                    arg,
                    self.parameters,
                    inputs=[],
                    outputs=[File(_new_file(self.context))], # for output logs
                    )

    @classmethod
    def create_apps(cls, context):
        assert not (cls == BaseReference) # should never be called directly
        app_length_dataset = context.apps(Dataset, 'length_dataset')
        app_save_dataset   = context.apps(Dataset, 'save_dataset')

        def evaluate_multiple(parameters, inputs=[], outputs=[]):
            data = []
            nstates = app_length_dataset(inputs=[inputs[0]])
            for i in range(nstates.result()):
                data.append(context.apps(cls, 'evaluate_single')(
                    read_dataset(i, inputs=[inputs[0]], outputs=[]),
                    parameters,
                    inputs=[],
                    outputs=[],
                    ))
            return app_save_dataset(states=None, inputs=data, outputs=[outputs[0]])
        app_evaluate_multiple = join_app(evaluate_multiple)
        context.register_app(cls, 'evaluate_multiple', app_evaluate_multiple)
