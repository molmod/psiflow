import os
import pytest
import numpy as np
from parsl.app.futures import DataFuture

from ase import Atoms
from ase.io import read, write
from ase.io.extxyz import write_extxyz

from psiflow.data import FlowAtoms, Dataset, NullState
from psiflow.utils import get_index_element_mask, is_reduced

from tests.conftest import generate_emt_cu_data # explicit import for regular function


def test_flow_atoms(context, dataset, tmp_path):
    atoms = dataset.get(index=0).result().copy() # copy necessary with HTEX!
    assert type(atoms) == FlowAtoms
    atoms.reference_status = True
    atoms_ = atoms.copy()
    assert atoms_.reference_status == True
    atoms_ = FlowAtoms.from_atoms(atoms)
    assert atoms_.reference_status == True
    for i in range(dataset.length().result()):
        atoms = dataset[i].result()
        assert type(atoms) == FlowAtoms
        assert atoms.reference_status == True
    assert dataset.labeled().length().result() == dataset.length().result()
    dataset += Dataset([NullState])
    assert dataset.length().result() == 1 + dataset.not_null().length().result()
    assert atoms.reference_status == True
    atoms.reset()
    atoms.cell[:] = np.array([
            [3, 1, 1],
            [1, 5, 0],
            [0, -1, 5]])
    assert not 'energy' in atoms.info
    assert atoms.reference_status == False
    assert tuple(sorted(atoms.elements)) == ('Cu', 'H')
    assert not is_reduced(atoms.cell)
    atoms.canonical_orientation()
    assert is_reduced(atoms.cell)


def test_dataset_empty(context, tmp_path):
    dataset = Dataset(atoms_list=[])
    assert dataset.length().result() == 0
    assert isinstance(dataset.data_future, DataFuture)
    path_xyz = tmp_path / 'test.xyz'
    dataset.save(path_xyz).result() # ensure the copy is executed before assert
    assert os.path.isfile(path_xyz)
    with pytest.raises(ValueError): # cannot save outside cwd
        dataset.save(path_xyz.parents[3] / 'test.xyz')


def test_dataset_append(dataset):
    assert 20 == dataset.length().result()
    atoms_list = dataset.as_list().result()
    assert len(atoms_list) == 20
    assert type(atoms_list) == list
    assert type(atoms_list[0]) == FlowAtoms
    empty = Dataset([]) # use [] instead of None
    empty.append(dataset)
    assert 20 == empty.length().result()
    dataset.append(dataset)
    assert 40 == dataset.length().result()
    added = dataset + dataset
    assert added.length().result() == 80
    assert dataset.length().result() == 40 # may not changed
    dataset += dataset
    assert dataset.length().result() == 80 # may not changed

    # test canonical transformation
    transformed = dataset.canonical_orientation()
    for i in range(dataset.length().result()):
        atoms = dataset[i].result()
        atoms.canonical_orientation()
        assert np.allclose(
                atoms.positions,
                transformed[i].result().positions,
                )
        assert np.allclose(
                atoms.cell,
                transformed[i].result().cell,
                )


def test_dataset_slice(dataset):
    part = dataset[:8]
    assert part.length().result() == 8
    state = dataset[8]
    data = Dataset(atoms_list=[state])
    assert data.length().result() == 1

    dataset_ = dataset.shuffle()
    equal = np.array([False] * dataset.length().result())
    for i in range(dataset.length().result()):
        equal[i] = np.allclose(
                dataset_[i].result().get_positions(),
                dataset[i].result().get_positions(),
                )
    assert not np.all(equal)


def test_dataset_from_xyz(context, tmp_path):
    data = generate_emt_cu_data(20, 0.1)
    path_xyz = tmp_path / 'data.xyz'
    with open(path_xyz, 'w') as f:
        write_extxyz(f, data)
    dataset = Dataset.load(path_xyz)

    for i in range(20):
        assert np.allclose(
                data[i].get_positions(),
                dataset[i].result().get_positions(),
                )


def test_dataset_metric(context, dataset):
    errors = Dataset.get_errors(dataset, None)
    assert errors.result().shape == (dataset.length().result(), 3)
    errors = np.mean(errors.result(), axis=0)
    assert np.all(errors > 0)
    with pytest.raises(AssertionError):
        errors = Dataset.get_errors(dataset, None, atom_indices=[0, 1])
        errors.result()
    with pytest.raises(AssertionError):
        errors = Dataset.get_errors(dataset, None, elements=['C'])
        errors.result()
    # should return empty array
    errors = Dataset.get_errors(dataset, None, elements=['Ne'], properties=['forces'])
    assert errors.result().shape == (0, 1)

    errors = Dataset.get_errors(dataset, None, elements=['H'], properties=['forces']) # H present
    errors.result()
    errors_rmse = Dataset.get_errors(dataset, None, elements=['Cu'], properties=['forces']) # Cu present
    errors_mae  = Dataset.get_errors(dataset, None, elements=['Cu'], properties=['forces'], metric='mae') # Cu present
    errors_max  = Dataset.get_errors(dataset, None, elements=['Cu'], properties=['forces'], metric='max') # Cu present
    assert np.all(errors_rmse.result() > errors_mae.result())
    assert np.all(errors_max.result() > errors_rmse.result())

    atoms = FlowAtoms(numbers=30 * np.ones(10), positions=np.zeros((10, 3)), pbc=False)
    atoms.info['forces'] = np.random.uniform(-1, 1, size=(10, 3))
    dataset.append(Dataset([atoms]))
    errors = Dataset.get_errors(dataset, None, elements=['H'], properties=['forces']) # H present
    assert errors.result().shape[0] == dataset.length().result() - 1


def test_index_element_mask():
    numbers = np.array([1, 1, 1, 6, 6, 8, 8, 8])
    elements = ['H']
    indices = [0, 1, 4, 5]
    assert np.allclose(
            get_index_element_mask(numbers, elements, indices),
            np.array([True, True, False, False, False, False, False, False])
            )
    elements = ['H', 'O']
    indices = [0, 1, 4, 5]
    assert np.allclose(
            get_index_element_mask(numbers, elements, indices),
            np.array([True, True, False, False, False, True, False, False])
            )
    elements = ['H', 'O']
    indices = [3]
    assert np.allclose(
            get_index_element_mask(numbers, elements, indices),
            np.array([False] * len(numbers))
            )
    elements = ['H', 'O']
    indices = None
    assert np.allclose(
            get_index_element_mask(numbers, elements, indices),
            np.array([True, True, True, False, False, True, True, True])
            )
    elements = None
    indices = [0, 1, 2, 3, 5]
    assert np.allclose(
            get_index_element_mask(numbers, elements, indices),
            np.array([True, True, True, True, False, True, False, False])
            )
    elements = ['Cl'] # not present at all
    indices = None
    assert np.allclose(
            get_index_element_mask(numbers, elements, indices),
            np.array([False] * len(numbers))
            )


def test_dataset_gather(context, dataset):
    indices = [0, 3, 2, 6, 1]
    gathered = dataset[indices]
    assert gathered.length().result() == len(indices)
    for i, index in enumerate(indices):
        assert np.allclose(
                dataset[index].result().positions,
                gathered[i].result().positions,
                )


def test_data_elements(context, dataset):
    assert 'H' in dataset.elements().result()
    assert 'Cu' in dataset.elements().result()
    assert len(dataset.elements().result()) == 2


def test_data_reset(context, dataset):
    assert tuple(dataset.energy_labels().result()) == ('energy',)
    dataset = dataset.reset()
    assert tuple(dataset.energy_labels().result()) == tuple()
    assert not 'energy' in dataset[0].result().info


def test_nullstate(context):
    state = FlowAtoms(
            numbers=np.array([0]),
            positions=np.array([[0.0, 0, 0]]),
            pbc=False,
            )
    assert not id(state) == id(NullState)
    assert state == NullState
