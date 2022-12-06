import tempfile

from parsl.app.app import python_app
from parsl.data_provider.files import File

from autolearn.execution import ModelExecutionDefinition
from autolearn.utils import copy_file


def save_dataset(data, outputs=[]):
    from ase.io.extxyz import write_extxyz
    with open(outputs[0], 'w') as f:
        write_extxyz(f, data)


def read_dataset(index=None, inputs=[]):
    from ase.io.extxyz import read_extxyz
    with open(inputs[0], 'r' ) as f:
        if index is None:
            data = list(read_extxyz(f, index=slice(None)))
        elif isinstance(index, slice):
            data = list(read_extxyz(f, index=index))
        else:
            data = list(read_extxyz(f, index=index))[0] # return as Atoms
    return data


def join_dataset(inputs=[], outputs=[]):
    data_0 = read_dataset(inputs=[inputs[0]])
    data_1 = read_dataset(inputs=[inputs[1]])
    save_dataset(data_0 + data_1, outputs=[outputs[0]])


def get_length_dataset(inputs=[]):
    data = read_dataset(inputs=[inputs[0]])
    return len(data)


class Dataset:
    """Base class for a dataset of atomic structures"""

    def __init__(self, context, data=[]):
        """Constructor

        Arguments
        ---------

        file : File or DataFuture object representing the dataset

        """
        self.context = context
        p_save_dataset = python_app(
                save_dataset,
                executors=[self.executor_label],
                )
        self.future = p_save_dataset(
                data,
                outputs=[File(self.new_xyz())],
                ).outputs[0]

    def new_xyz(self):
        _, name = tempfile.mkstemp(
                suffix='.xyz',
                prefix='data_',
                dir=self.context.path,
                )
        return name

    def save(self, path_dataset):
        p_copy_file = python_app(copy_file, executors=[self.executor_label])
        return p_copy_file(inputs=[self.future], outputs=[File(str(path_dataset))])

    def __add__(self, dataset):
        dataset_sum = Dataset(self.context, data=[])
        p_join_dataset = python_app(
                join_dataset,
                executors=[self.executor_label],
                )
        dataset_sum.future = p_join_dataset(
                inputs=[self.future, dataset.future],
                outputs=[File(dataset_sum.new_xyz())],
                ).outputs[0]
        return dataset_sum

    @property
    def executor_label(self):
        return self.context[ModelExecutionDefinition].executor_label

    def length(self):
        p_get_length_dataset = python_app(
                get_length_dataset,
                executors=[self.executor_label],
                )
        return p_get_length_dataset(inputs=[self.future])

    def __getitem__(self, i):
        p_read_dataset = python_app(
                read_dataset,
                executors=[self.executor_label],
                )
        data = p_read_dataset(index=i, inputs=[self.future])
        if isinstance(i, slice):
            return Dataset(self.context, data)
        else:
            return data # represents an AppFuture of an ase.Atoms instance

    @classmethod
    def from_xyz(cls, context, path_xyz):
        dataset = cls(context, data=[])
        dataset.future = File(str(path_xyz))
        return dataset
