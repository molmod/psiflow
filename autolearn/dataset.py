import os
import tempfile

from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from autolearn.execution import ModelExecutionDefinition, Container
from autolearn.utils import copy_file


def save_dataset(states, inputs=[], outputs=[]):
    from ase.io.extxyz import write_extxyz
    if states is not None:
        _data = states
    else:
        _data = inputs
    with open(outputs[0], 'w') as f:
        write_extxyz(f, _data)


def read_dataset(index, inputs=[], outputs=[]):
    from ase.io.extxyz import read_extxyz, write_extxyz
    with open(inputs[0], 'r' ) as f:
        data = list(read_extxyz(f, index=index))
    if isinstance(index, int): # unpack list to single Atoms object
        assert len(data) == 1
        data = data[0]
    if len(outputs) > 0: # save to file
        with open(outputs[0], 'w') as f:
            write_extxyz(f, data)
    return data


def join_dataset(inputs=[], outputs=[]):
    data = []
    for i in range(len(inputs)):
        data += read_dataset(slice(None), inputs=[inputs[i]]) # read all
    save_dataset(data, outputs=[outputs[0]])


def get_length_dataset(inputs=[]):
    data = read_dataset(slice(None), inputs=[inputs[0]])
    return len(data)


def _new_xyz(context):
    _, name = tempfile.mkstemp(
            suffix='.xyz',
            prefix='data_',
            dir=context.path,
            )
    return name


class Dataset(Container):
    """Container to represent a dataset of atomic structures"""

    def __init__(self, context, atoms_list=[], data=None):
        """Constructor

        Arguments
        ---------

        context : ExecutionContext

        atoms_list : list of Atoms objects

        data : DataFuture

        """
        super().__init__(context) # create apps

        if data is None: # generate new DataFuture
            if isinstance(atoms_list, AppFuture):
                states = atoms_list
                inputs = []
            else:
                if (len(atoms_list) > 0) and isinstance(atoms_list[0], AppFuture):
                    states = None
                    inputs = atoms_list
                else:
                    states = atoms_list
                    inputs = []
            self.data = self.apps['save_dataset'](
                    states,
                    inputs=inputs,
                    outputs=[File(str(_new_xyz(context)))],
                    ).outputs[0]
        else:
            assert len(atoms_list) == 0 # do not allow additional atoms
            assert (isinstance(data, DataFuture) or isinstance(data, File))
            self.data = data

    def save(self, path_dataset):
        return self.apps['copy_dataset'](
                inputs=[self.data],
                outputs=[File(str(path_dataset))],
                )

    def length(self):
        return self.apps['length_dataset'](inputs=[self.data])

    def __getitem__(self, i):
        if isinstance(i, slice):
            data = self.apps['read_dataset'](
                    index=i,
                    inputs=[self.data],
                    outputs=[File(_new_xyz(self.context))],
                    ).outputs[0]
            return Dataset(self.context, data=data)
        else:
            return self.apps['read_dataset'](
                    index=i,
                    inputs=[self.data],
                    ) # represents an AppFuture of an ase.Atoms instance

    @classmethod
    def from_xyz(cls, context, path_xyz):
        assert os.path.isfile(path_xyz) # needs to be locally accessible
        return cls(context, data=File(str(path_xyz)))

    @staticmethod
    def merge(*datasets):
        assert len(datasets) > 0
        context = datasets[0].context
        apps = Dataset.create_apps(context)
        data = apps['join_dataset'](
                inputs=[item.data for item in datasets],
                outputs=[File(_new_xyz(context))],
                ).outputs[0]
        return Dataset(context, data=data)

    @staticmethod
    def create_apps(context):
        executor_label = context[ModelExecutionDefinition].executor_label
        apps = {}
        apps['save_dataset'] = python_app(
                save_dataset,
                executors=[executor_label],
                )
        apps['read_dataset'] = python_app(
                read_dataset,
                executors=[executor_label],
                )
        apps['copy_dataset'] = python_app(
                copy_file,
                executors=[executor_label],
                )
        apps['join_dataset'] = python_app(
                join_dataset,
                executors=[executor_label],
                )
        apps['length_dataset'] = python_app(
                get_length_dataset,
                executors=[executor_label],
                )
        return apps
