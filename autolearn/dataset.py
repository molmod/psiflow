import os
import tempfile

from parsl.app.app import python_app, join_app
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


def compute_metrics(
        intrinsic,
        atom_indices,
        elements,
        metric,
        properties,
        inputs=[],
        ):
    import numpy as np
    from autolearn.dataset import read_dataset
    from autolearn.utils import get_index_element_mask
    data = read_dataset(slice(None), inputs=[inputs[0]])
    errors = np.zeros((len(data), len(properties)))
    outer_mask = np.array([True] * len(data))
    for i, atoms in enumerate(data):
        if (atom_indices is not None) or (elements is not None):
            assert 'energy' not in properties
            assert 'stress' not in properties
            assert 'forces' in properties # only makes sense for forces
            mask = get_index_element_mask(atoms.numbers, elements, atom_indices)
        else:
            mask = np.array([True] * len(atoms))
        if not np.any(mask): # no target atoms present; skip
            outer_mask[i] = False
            continue
        if not intrinsic:
            if 'energy' in properties:
                assert 'energy_model' in atoms.info.keys()
            if 'forces' in properties:
                assert 'forces_model' in atoms.arrays.keys()
            if 'stress' in properties:
                assert 'stress_model' in atoms.info.keys()
        for j, property_ in enumerate(properties):
            if intrinsic:
                atoms.info['energy_model'] = 0.0
                atoms.arrays['forces_model'] = np.zeros(atoms.positions.shape)
                atoms.info['stress_model'] = np.zeros((3, 3))
            if property_ == 'energy':
                array_0 = np.array([atoms.info['energy']]).reshape((1, 1))
                array_1 = np.array([atoms.info['energy_model']]).reshape((1, 1))
            elif property_ == 'forces':
                array_0 = atoms.arrays['forces'][mask, :]
                array_1 = atoms.arrays['forces_model'][mask, :]
            elif property_ == 'stress':
                array_0 = atoms.info['stress'].reshape((1, 9))
                array_1 = atoms.info['stress_model'].reshape((1, 9))
            else:
                raise ValueError('property {} unknown!'.format(property_))
            if metric == 'mae':
                errors[i, j] = np.mean(np.abs(array_0 - array_1))
            elif metric == 'rmse':
                errors[i, j] = np.mean(np.linalg.norm(array_0 - array_1, axis=1))
            else:
                raise ValueError('metric {} unknown!'.format(metric))
    if not np.any(outer_mask):
        raise AssertionError('no states in dataset contained atoms of interest')
    return errors[outer_mask, :]


class Dataset(Container):
    """Container to represent a dataset of atomic structures"""

    def __init__(self, context, atoms_list=[], data_future=None):
        """Constructor

        Arguments
        ---------

        context : ExecutionContext

        atoms_list : list of Atoms objects

        data_future : DataFuture

        """
        super().__init__(context)

        if data_future is None: # generate new DataFuture
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
            self.data_future = context.apps(Dataset, 'save_dataset')(
                    states,
                    inputs=inputs,
                    outputs=[File(str(_new_xyz(context)))],
                    ).outputs[0]
        else:
            assert len(atoms_list) == 0 # do not allow additional atoms
            assert (isinstance(data_future, DataFuture) or isinstance(data_future, File))
            self.data_future = data_future

    def save(self, path_dataset):
        return self.context.apps(Dataset, 'copy_dataset')(
                inputs=[self.data_future],
                outputs=[File(str(path_dataset))],
                )

    def length(self):
        return self.context.apps(Dataset, 'length_dataset')(inputs=[self.data_future])

    def __getitem__(self, i):
        if isinstance(i, slice):
            data_future = self.context.apps(Dataset, 'read_dataset')(
                    index=i,
                    inputs=[self.data_future],
                    outputs=[File(_new_xyz(self.context))],
                    ).outputs[0]
            return Dataset(self.context, data_future=data_future)
        else:
            return self.context.apps(Dataset, 'read_dataset')(
                    index=i,
                    inputs=[self.data_future],
                    ) # represents an AppFuture of an ase.Atoms instance

    def get_errors(
            self,
            intrinsic=False,
            atom_indices=None,
            elements=None,
            metric='rmse',
            properties=['energy', 'forces', 'stress'],
            ):
        return self.context.apps(Dataset, 'compute_metrics')(
                intrinsic=intrinsic,
                atom_indices=atom_indices,
                elements=elements,
                metric=metric,
                properties=properties,
                inputs=[self.data_future],
                )

    @classmethod
    def from_xyz(cls, context, path_xyz):
        assert os.path.isfile(path_xyz) # needs to be locally accessible
        return cls(context, data_future=File(str(path_xyz)))

    @staticmethod
    def merge(*datasets):
        assert len(datasets) > 0
        context = datasets[0].context
        data_future = context.apps(Dataset, 'join_dataset')(
                inputs=[item.data_future for item in datasets],
                outputs=[File(_new_xyz(context))],
                ).outputs[0]
        return Dataset(context, data_future=data_future)

    @staticmethod
    def create_apps(context):
        executor_label = context[ModelExecutionDefinition].executor_label
        app_save_dataset = python_app(save_dataset, executors=[executor_label])
        context.register_app(Dataset, 'save_dataset', app_save_dataset)

        app_read_dataset = python_app(read_dataset, executors=[executor_label])
        context.register_app(Dataset, 'read_dataset', app_read_dataset)

        app_copy_dataset = python_app(copy_file, executors=[executor_label])
        context.register_app(Dataset, 'copy_dataset', app_copy_dataset)

        app_join_dataset = python_app(join_dataset, executors=[executor_label])
        context.register_app(Dataset, 'join_dataset', app_join_dataset)

        app_length_dataset = python_app(get_length_dataset, executors=[executor_label])
        context.register_app(Dataset, 'length_dataset', app_length_dataset)

        app_compute_metrics = python_app(compute_metrics, executors=[executor_label])
        context.register_app(Dataset, 'compute_metrics', app_compute_metrics)
