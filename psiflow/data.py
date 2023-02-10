"""
The `data` module implements objects used in the representation and IO of
atomic data.
An atomic configuration is defined by the number and cartesian coordinate of
each of its atoms as well as three noncoplanar box vectors which define the periodicity
of the system.
In addition, an atomic configuration may be *labeled* with the total potential
energy of the configuration, the atomic forces, and the virial stress tensor.
Finally, it can also contain pointers to
the output and error logs of QM evaluation calculations.

"""

from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List
import typeguard
import copy
import os
import tempfile
import logging
import numpy as np
from pathlib import Path

from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.dataflow.memoization import id_for_memo

from ase import Atoms

from psiflow.execution import Container, ExecutionContext
from psiflow.utils import copy_data_future, copy_app_future


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


@typeguard.typechecked
class FlowAtoms(Atoms):
    """Wrapper class around ASE `Atoms` with additional attributes for QM logs

    In addition to the standard `Atoms` functionality, this class offers the
    ability to store pointers to output and error logs that have been generated
    during a QM evaluation of the atomic structure. A separate attribute
    is reserved to store the exit code of the calculation (success or failed)
    as a boolean.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if 'reference_stdout' not in self.info.keys(): # only set if not present
            self.info['reference_stdout'] = False # default None not supported
        if 'reference_stderr' not in self.info.keys(): # only set if not present
            self.info['reference_stderr'] = False
        if 'reference_status' not in self.info.keys(): # only set if not present
            self.info['reference_status'] = False

    @property
    def reference_status(self) -> bool:
        """True if QM evaluation was successful, False otherwise"""
        return self.info['reference_status']

    @reference_status.setter
    def reference_status(self, flag: bool) -> None:
        assert flag in [True, False]
        self.info['reference_status'] = flag

    @property
    def reference_stdout(self) -> Union[bool, str]:
        """Contains filepath to QM output log, False if not yet performed"""
        return self.info['reference_stdout']

    @reference_stdout.setter
    def reference_stdout(self, path: Union[bool, str]) -> None:
        self.info['reference_stdout'] = path

    @property
    def reference_stderr(self) -> Union[bool, str]:
        """Contains filepath to QM error log, False if not yet performed"""
        return self.info['reference_stderr']

    @reference_stderr.setter
    def reference_stderr(self, path: Union[bool, str]) -> None:
        self.info['reference_stderr'] = path

    def copy(self) -> FlowAtoms:
        """Performs a deepcopy of `self`"""
        flow_atoms = FlowAtoms.from_atoms(self)
        flow_atoms.reference_stdout = self.reference_stdout
        flow_atoms.reference_stderr = self.reference_stderr
        flow_atoms.reference_status = self.reference_status
        if 'stress' in flow_atoms.info.keys(): # bug in ASE constructor!
            flow_atoms.info['stress'] = flow_atoms.info['stress'].copy()
        return flow_atoms

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> FlowAtoms:
        """Generates a `FlowAtoms` object based on an existing `Atoms`

        Array attributes need to be copied manually as this is for some reason
        not done by the ASE constructor.

        Args:
            atoms (Atoms):
                contains atomic configuration to be stored as `FlowAtoms`

        """
        flow_atoms = FlowAtoms( # follows Atoms.copy method
                cell=atoms.cell,
                pbc=atoms.pbc,
                info=atoms.info,
                celldisp=atoms._celldisp.copy(),
                )
        flow_atoms.arrays = {}
        for name, a in atoms.arrays.items():
            flow_atoms.arrays[name] = a.copy()
        flow_atoms.constraints = copy.deepcopy(atoms.constraints)
        return flow_atoms


@id_for_memo.register(FlowAtoms)
def id_for_memo_flowatoms(atoms: FlowAtoms, output_ref=False):
    """Returns unique bytestring of instance used by Parsl for caching

    Args:
        atoms (FlowAtoms): object for which to construct a bytestring
        output_ref (bool): whether or not it is part of the app outputs

    """
    assert not output_ref
    string = ''
    string += str(atoms.numbers)
    string += str(atoms.cell.round(decimals=4))
    string += str(atoms.positions.round(decimals=4))
    return bytes(string, 'utf-8')


@typeguard.typechecked
def save_dataset(
        states: Optional[List[Optional[FlowAtoms]]],
        inputs: List[Optional[FlowAtoms]] = [], # allow None
        return_data: bool = False, # whether to return data
        outputs: List[File] = [],
        ) -> Optional[List[FlowAtoms]]:
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
    if return_data:
        return _data
app_save_dataset = python_app(save_dataset, executors=['default'])


@typeguard.typechecked
def _save_atoms(atoms: FlowAtoms, outputs=[]):
    from ase.io import write
    write(outputs[0].filepath, atoms)
save_atoms = python_app(_save_atoms, executors=['default'])


@typeguard.typechecked
def read_dataset(
        index_or_indices: Union[int, List[int], slice],
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> Union[FlowAtoms, List[FlowAtoms]]:
    from ase.io.extxyz import read_extxyz, write_extxyz
    from psiflow.data import FlowAtoms
    with open(inputs[0], 'r' ) as f:
        if type(index_or_indices) == int:
            atoms = list(read_extxyz(f, index=index_or_indices))[0]
            data  = FlowAtoms.from_atoms(atoms) # single atoms instance
        else:
            if type(index_or_indices) == list:
                data = [list(read_extxyz(f, index=i))[0] for i in index_or_indices]
            elif type(index_or_indices) == slice:
                data = list(read_extxyz(f, index=index_or_indices))
            else:
                raise ValueError
            data = [FlowAtoms.from_atoms(a) for a in data] # list of atoms
    if len(outputs) > 0: # save to file
        with open(outputs[0], 'w') as f:
            write_extxyz(f, data)
    return data
app_read_dataset = python_app(read_dataset, executors=['default'])


@typeguard.typechecked
def join_dataset(inputs: List[File] = [], outputs: List[File] = []) -> None:
    data = []
    for i in range(len(inputs)):
        data += read_dataset(slice(None), inputs=[inputs[i]]) # read all
    save_dataset(data, outputs=[outputs[0]])
app_join_dataset = python_app(join_dataset, executors=['default'])


@typeguard.typechecked
def get_length_dataset(inputs: List[File] = []) -> int:
    data = read_dataset(slice(None), inputs=[inputs[0]])
    return len(data)
app_length_dataset = python_app(get_length_dataset, executors=['default'])


@typeguard.typechecked
def get_indices_per_flag(
        flag: bool,
        inputs: List[File] = [],
        ) -> List[int]:
    data = read_dataset(slice(None), inputs=[inputs[0]])
    indices = []
    for i, atoms in enumerate(data):
        assert atoms.reference_status is not None
        if atoms.reference_status == flag:
            indices.append(i)
    return indices
app_get_indices = python_app(get_indices_per_flag, executors=['default'])


@typeguard.typechecked
def compute_metrics(
        intrinsic: bool,
        atom_indices: Optional[List[int]],
        elements: Optional[List[str]],
        metric: str,
        properties: List[str],
        inputs: List[File] = [],
        ) -> np.ndarray:
    import numpy as np
    from ase.units import Pascal
    from psiflow.data import read_dataset
    from psiflow.utils import get_index_element_mask
    data_0 = read_dataset(slice(None), inputs=[inputs[0]])
    if len(inputs) == 1:
        assert intrinsic
        data_1 = [a.copy() for a in data_0]
        for atoms_1 in data_1:
            if 'energy' in atoms_1.info.keys():
                atoms_1.info['energy'] = 0.0
            if 'stress' in atoms_1.info.keys(): # ASE copy fails for info attrs!
                atoms_1.info['stress'] = np.zeros((3, 3))
            if 'forces' in atoms_1.arrays.keys():
                atoms_1.arrays['forces'][:] = 0.0
    else:
        data_1 = read_dataset(slice(None), inputs=[inputs[1]])
    assert len(data_0) == len(data_1)
    for atoms_0, atoms_1 in zip(data_0, data_1):
        assert np.allclose(atoms_0.numbers, atoms_1.numbers)
        assert np.allclose(atoms_0.positions, atoms_1.positions)
        if atoms_0.cell is not None:
            assert np.allclose(atoms_0.cell, atoms_1.cell)

    errors = np.zeros((len(data_0), len(properties)))
    outer_mask = np.array([True] * len(data_0))
    for i in range(len(data_0)):
        atoms_0 = data_0[i]
        atoms_1 = data_1[i]
        if (atom_indices is not None) or (elements is not None):
            assert 'energy' not in properties
            assert 'stress' not in properties
            assert 'forces' in properties # only makes sense for forces
            mask = get_index_element_mask(atoms_0.numbers, elements, atom_indices)
        else:
            mask = np.array([True] * len(atoms_0))
        if not np.any(mask): # no target atoms present; skip
            outer_mask[i] = False
            continue
        if 'energy' in properties:
            assert 'energy' in atoms_0.info.keys()
            assert 'energy' in atoms_1.info.keys()
        if 'forces' in properties:
            assert 'forces' in atoms_0.arrays.keys()
            assert 'forces' in atoms_1.arrays.keys()
        if 'stress' in properties:
            assert 'stress' in atoms_0.info.keys()
            assert 'stress' in atoms_1.info.keys()
        for j, property_ in enumerate(properties):
            if property_ == 'energy':
                array_0 = np.array([atoms_0.info['energy']]).reshape((1, 1))
                array_1 = np.array([atoms_1.info['energy']]).reshape((1, 1))
                array_0 /= len(atoms_0) # per atom energy error
                array_1 /= len(atoms_1)
                array_0 *= 1000 # in meV/atom
                array_1 *= 1000
            elif property_ == 'forces':
                array_0 = atoms_0.arrays['forces'][mask, :]
                array_1 = atoms_1.arrays['forces'][mask, :]
                array_0 *= 1000 # in meV/angstrom
                array_1 *= 1000
            elif property_ == 'stress':
                array_0 = atoms_0.info['stress'].reshape((1, 9))
                array_1 = atoms_1.info['stress'].reshape((1, 9))
                array_0 /= (1e6 * Pascal) # in MPa
                array_1 /= (1e6 * Pascal)
            else:
                raise ValueError('property {} unknown!'.format(property_))
            if metric == 'mae':
                errors[i, j] = np.mean(np.abs(array_0 - array_1))
            elif metric == 'rmse':
                errors[i, j] = np.sqrt(np.mean((array_0 - array_1) ** 2))
            elif metric == 'max':
                errors[i, j] = np.max(np.linalg.norm(array_0 - array_1, axis=1))
            else:
                raise ValueError('metric {} unknown!'.format(metric))
    if not np.any(outer_mask):
        raise AssertionError('no states in dataset contained atoms of interest')
    return errors[outer_mask, :]
app_compute_metrics = python_app(compute_metrics, executors=['default'])


@typeguard.typechecked
class Dataset(Container):
    """Container to represent a dataset of atomic structures

    Args:
        context: an `ExecutionContext` instance with a 'default' executor.
        atoms_list: a list of `Atoms` instances which represent the dataset.
        data_future: a `parsl.app.futures.DataFuture` instance that points
            to an `.xyz` file.

    """
    execution_definition = ['executor']

    def __init__(
            self,
            context: ExecutionContext,
            atoms_list: Optional[Union[List[AppFuture], List[Union[FlowAtoms, Atoms]], AppFuture]],
            data_future: Optional[Union[DataFuture, File]] = None,
            ) -> None:
        super().__init__(context)

        if data_future is None: # generate new DataFuture
            assert atoms_list is not None
            if isinstance(atoms_list, AppFuture):
                states = atoms_list
                inputs = []
            else:
                if (len(atoms_list) > 0) and isinstance(atoms_list[0], AppFuture):
                    states = None
                    inputs = atoms_list
                else:
                    states = [FlowAtoms.from_atoms(a) for a in atoms_list]
                    inputs = []
            self.data_future = app_save_dataset(
                    states,
                    inputs=inputs,
                    outputs=[context.new_file('data_', '.xyz')],
                    ).outputs[0]
        else:
            assert atoms_list is None # do not allow additional atoms
            self.data_future = copy_data_future(
                    inputs=[data_future],
                    outputs=[context.new_file('data_', '.xyz')],
                    ).outputs[0] # ensure type(data_future) == DataFuture

    def length(self) -> AppFuture:
        return app_length_dataset(inputs=[self.data_future])

    def shuffle(self):
        indices = np.arange(self.length().result())
        np.random.shuffle(indices)
        return self.get(indices=[int(i) for i in indices])

    def __getitem__(
            self,
            index: Union[int, slice, List[int], AppFuture],
            ) -> Union[Dataset, AppFuture]:
        if isinstance(index, int):
            return self.get(index=index)
        else: # slice, List, AppFuture
            return self.get(indices=index)

    def get(
            self,
            index: Optional[int] = None,
            indices: Optional[Union[List[int], AppFuture, slice]] = None,
            ) -> Union[Dataset, AppFuture]:
        if indices is not None:
            assert index is None
            data_future = app_read_dataset(
                    indices,
                    inputs=[self.data_future],
                    outputs=[self.context.new_file('data_', '.xyz')],
                    ).outputs[0]
            return Dataset(self.context, None, data_future=data_future)
        else:
            assert index is not None
            atoms = app_read_dataset(
                    index, # int or AppFuture of int
                    inputs=[self.data_future],
                    ) # represents an AppFuture of an ase.Atoms instance
            return atoms

    def save(
            self,
            path_dataset: Union[Path, str],
            require_done: bool = True,
            ) -> AppFuture:
        future = copy_data_future(
                inputs=[self.data_future],
                outputs=[File(str(path_dataset))],
                )
        if require_done:
            future.result()
        return future

    def as_list(self) -> List[FlowAtoms]:
        return app_read_dataset(
                index_or_indices=slice(None),
                inputs=[self.data_future],
                ).result()

    def append(self, dataset: Dataset) -> None:
        self.data_future = app_join_dataset(
                inputs=[self.data_future, dataset.data_future],
                outputs=[self.context.new_file('data_', '.xyz')],
                ).outputs[0]

    def __add__(self, dataset: Dataset) -> Dataset:
        data_future = app_join_dataset(
                inputs=[self.data_future, dataset.data_future],
                outputs=[self.context.new_file('data_', '.xyz')],
                ).outputs[0]
        return Dataset(self.context, None, data_future)

    def log(self, name):
        logger.info('dataset {} contains {} states'.format(name, self.length().result()))

    @property
    def success(self) -> AppFuture:
        return app_get_indices(
                True,
                inputs=[self.data_future],
                )

    @property
    def failed(self) -> AppFuture:
        return app_get_indices(
                False,
                inputs=[self.data_future],
                )

    @staticmethod
    def get_errors(
            dataset_0: Dataset,
            dataset_1: Optional[Dataset], # None when computing intrinsic errors
            atom_indices: Optional[List[int]] = None,
            elements: Optional[List[str]] = None,
            metric: str = 'rmse',
            properties: List[str] = ['energy', 'forces', 'stress'],
            ) -> AppFuture:
        inputs = [dataset_0.data_future]
        if dataset_1 is not None:
            inputs.append(dataset_1.data_future)
            intrinsic = False
        else:
            intrinsic = True
        return app_compute_metrics(
                intrinsic=intrinsic,
                atom_indices=atom_indices,
                elements=elements,
                metric=metric,
                properties=properties,
                inputs=inputs,
                )

    @classmethod
    def load(
            cls,
            context: ExecutionContext,
            path_xyz: Union[Path, str],
            ) -> Dataset:
        assert os.path.isfile(path_xyz) # needs to be locally accessible
        return cls(context, None, data_future=File(str(path_xyz)))

    @staticmethod
    def create_apps(context: ExecutionContext) -> None:
        pass # no apps beyond default executor
