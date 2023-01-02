import os
import tempfile

from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from ase import Atoms

from flower.execution import Container
from flower.utils import copy_data_future, _new_file


class FlowerAtoms(Atoms):
    """Wrapper class around ase Atoms with additional attributes for QM logs"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_log   = None
        self.info['evaluation_flag'] = None

    @property
    def evaluation_flag(self):
        return self.info['evaluation_flag']

    @evaluation_flag.setter
    def evaluation_flag(self, flag):
        assert flag in [None, 'success', 'failed']
        self.info['evaluation_flag'] = flag

    @classmethod
    def from_atoms(cls, atoms):
        flower_atoms = cls(
                numbers=atoms.numbers,
                positions=atoms.get_positions(),
                cell=atoms.get_cell(),
                pbc=atoms.pbc,
                )
        properties = ['energy', 'stress']
        for property_ in properties:
            for key in atoms.info.keys():
                if key.startswith(property_):
                    flower_atoms.info[key] = atoms.info[key]
        properties = ['forces']
        for property_ in properties:
            for key in atoms.arrays.keys():
                if key.startswith(property_):
                    flower_atoms.arrays[key] = atoms.arrays[key]

        if 'evaluation_flag' in atoms.info.keys():
            # default ASE value is True; should be converted to None
            value = atoms.info['evaluation_flag']
            if value == True:
                value = None
            flower_atoms.evaluation_flag = value
        return flower_atoms


def save_dataset(states, inputs=[], outputs=[]):
    from ase.io.extxyz import write_extxyz
    if states is not None:
        _data = states
    else:
        _data = inputs
    i = 0
    while i < len(_data):
        if _data[i] is None:
            del _data[i]
        else:
            i += 1
    with open(outputs[0], 'w') as f:
        write_extxyz(f, _data)


def read_dataset(index_or_indices, inputs=[], outputs=[]):
    from ase.io.extxyz import read_extxyz, write_extxyz
    from flower.data import FlowerAtoms
    with open(inputs[0], 'r' ) as f:
        if type(index_or_indices) == int:
            atoms = list(read_extxyz(f, index=index_or_indices))[0]
            data  = FlowerAtoms.from_atoms(atoms) # single atoms instance
        else:
            if type(index_or_indices) == list:
                data = [list(read_extxyz(f, index=i))[0] for i in index_or_indices]
            elif type(index_or_indices) == slice:
                data = list(read_extxyz(f, index=index_or_indices))
            else:
                raise ValueError
            data = [FlowerAtoms.from_atoms(a) for a in data] # list of atoms
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


def get_indices_per_flag(flag, inputs=[]):
    data = read_dataset(slice(None), inputs=[inputs[0]])
    indices = []
    for i, atoms in enumerate(data):
        assert atoms.evaluation_flag is not None
        if atoms.evaluation_flag == flag:
            indices.append(i)
    return indices


def compute_metrics(
        intrinsic,
        atom_indices,
        elements,
        metric,
        properties,
        suffix_0,
        suffix_1,
        inputs=[],
        ):
    import numpy as np
    from flower.data import read_dataset
    from flower.utils import get_index_element_mask
    data = read_dataset(slice(None), inputs=[inputs[0]])
    errors = np.zeros((len(data), len(properties)))
    outer_mask = np.array([True] * len(data))
    assert suffix_0 != suffix_1
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
        if 'energy' in properties:
            assert 'energy' + suffix_0 in atoms.info.keys()
            if not intrinsic:
                assert 'energy' + suffix_1 in atoms.info.keys()
        if 'forces' in properties:
            assert 'forces' + suffix_0 in atoms.arrays.keys()
            if not intrinsic:
                assert 'forces' + suffix_1 in atoms.arrays.keys()
        if 'stress' in properties:
            assert 'stress' + suffix_0 in atoms.info.keys()
            if not intrinsic:
                assert 'stress' + suffix_1 in atoms.info.keys()
        for j, property_ in enumerate(properties):
            if intrinsic:
                atoms.info['energy' + suffix_1] = 0.0
                atoms.info['stress' + suffix_1] = np.zeros((3, 3))
                atoms.arrays['forces' + suffix_1] = np.zeros(atoms.positions.shape)
            if property_ == 'energy':
                array_0 = np.array([atoms.info['energy' + suffix_0]]).reshape((1, 1))
                array_1 = np.array([atoms.info['energy' + suffix_1]]).reshape((1, 1))
            elif property_ == 'forces':
                array_0 = atoms.arrays['forces' + suffix_0][mask, :]
                array_1 = atoms.arrays['forces' + suffix_1][mask, :]
            elif property_ == 'stress':
                array_0 = atoms.info['stress' + suffix_0].reshape((1, 9))
                array_1 = atoms.info['stress' + suffix_1].reshape((1, 9))
            else:
                raise ValueError('property {} unknown!'.format(property_))
            if metric == 'mae':
                errors[i, j] = np.mean(np.abs(array_0 - array_1))
            elif metric == 'rmse':
                errors[i, j] = np.mean(np.linalg.norm(array_0 - array_1, axis=1))
            elif metric == 'max':
                errors[i, j] = np.max(np.linalg.norm(array_0 - array_1, axis=1))
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
                    states = [FlowerAtoms.from_atoms(a) for a in atoms_list]
                    inputs = []
            path_new = _new_file(context.path, prefix='data_', suffix='.xyz')
            self.data_future = context.apps(Dataset, 'save_dataset')(
                    states,
                    inputs=inputs,
                    outputs=[File(path_new)],
                    ).outputs[0]
        else:
            assert len(atoms_list) == 0 # do not allow additional atoms
            assert (isinstance(data_future, DataFuture) or isinstance(data_future, File))
            self.data_future = data_future

    def length(self):
        return self.context.apps(Dataset, 'length_dataset')(inputs=[self.data_future])

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get(index=index)
        else: # slice, List, AppFuture
            return self.get(indices=index)

    def get(self, index=None, indices=None):
        if indices is not None:
            assert index is None
            path_new = _new_file(self.context.path, 'data_', '.xyz')
            data_future = self.context.apps(Dataset, 'read_dataset')(
                    indices,
                    inputs=[self.data_future],
                    outputs=[File(path_new)],
                    ).outputs[0]
            return Dataset(self.context, data_future=data_future)
        else:
            assert index is not None
            return self.context.apps(Dataset, 'read_dataset')(
                    index, # int or AppFuture of int
                    inputs=[self.data_future],
                    ) # represents an AppFuture of an ase.Atoms instance

    def get_errors(
            self,
            intrinsic=False,
            atom_indices=None,
            elements=None,
            metric='rmse',
            suffix_0='', # use QM reference by default
            suffix_1='_model', # use single model by default 
            properties=['energy', 'forces', 'stress'],
            ):
        return self.context.apps(Dataset, 'compute_metrics')(
                intrinsic=intrinsic,
                atom_indices=atom_indices,
                elements=elements,
                metric=metric,
                properties=properties,
                suffix_0=suffix_0,
                suffix_1=suffix_1,
                inputs=[self.data_future],
                )

    def save(self, path_dataset, require_done=True):
        future = copy_data_future(
                inputs=[self.data_future],
                outputs=[File(str(path_dataset))],
                )
        if require_done:
            future.result()
        return future

    def append(self, dataset):
        path_new = _new_file(self.context.path, prefix='data_', suffix='.xyz')
        self.data_future = self.context.apps(Dataset, 'join_dataset')(
                inputs=[self.data_future, dataset.data_future],
                outputs=[File(path_new)],
                ).outputs[0]

    @property
    def success(self):
        return self.context.apps(Dataset, 'get_indices_per_flag')(
                'success',
                inputs=[self.data_future],
                )

    @property
    def failed(self):
        return self.context.apps(Dataset, 'get_indices_per_flag')(
                'failed',
                inputs=[self.data_future],
                )

    @classmethod
    def load(cls, context, path_xyz):
        assert os.path.isfile(path_xyz) # needs to be locally accessible
        return cls(context, data_future=File(str(path_xyz)))

    @staticmethod
    def merge(*datasets):
        assert len(datasets) > 0
        context = datasets[0].context
        path_new = _new_file(context.path, prefix='data_', suffix='.xyz')
        data_future = context.apps(Dataset, 'join_dataset')(
                inputs=[item.data_future for item in datasets],
                outputs=[File(path_new)],
                ).outputs[0]
        return Dataset(context, data_future=data_future)

    @staticmethod
    def create_apps(context):
        label = 'default'
        app_save_dataset = python_app(save_dataset, executors=[label])
        context.register_app(Dataset, 'save_dataset', app_save_dataset)

        app_read_dataset = python_app(read_dataset, executors=[label])
        context.register_app(Dataset, 'read_dataset', app_read_dataset)

        app_join_dataset = python_app(join_dataset, executors=[label])
        context.register_app(Dataset, 'join_dataset', app_join_dataset)

        app_length_dataset = python_app(get_length_dataset, executors=[label])
        context.register_app(Dataset, 'length_dataset', app_length_dataset)

        app_get_indices = python_app(get_indices_per_flag, executors=[label])
        context.register_app(Dataset, 'get_indices_per_flag', app_get_indices)

        app_compute_metrics = python_app(compute_metrics, executors=[label])
        context.register_app(Dataset, 'compute_metrics', app_compute_metrics)
