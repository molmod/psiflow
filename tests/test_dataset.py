import os
import pytest
import numpy as np
from parsl.app.futures import DataFuture

from ase import Atoms
from ase.io.extxyz import write_extxyz

from flower.data import FlowerAtoms, Dataset
from flower.utils import get_index_element_mask

from tests.conftest import generate_emt_cu_data # explicit import for regular function


def test_flower_atoms(context, dataset):
    for i in range(dataset.length().result()):
        atoms = dataset[i].result()
        assert isinstance(atoms, FlowerAtoms)
        assert atoms.evaluation_log is None
        assert atoms.evaluation_flag is None


def test_dataset_empty(context, tmp_path):
    dataset = Dataset(context, atoms_list=[])
    assert dataset.length().result() == 0
    assert isinstance(dataset.data_future, DataFuture)
    path_xyz = tmp_path / 'test.xyz'
    dataset.save(path_xyz).result() # ensure the copy is executed before assert
    assert os.path.isfile(path_xyz)


def test_dataset_append(dataset):
    assert 20 == dataset.length().result()
    empty = Dataset(dataset.context)
    empty.append(dataset)
    assert 20 == empty.length().result()
    new = Dataset.merge(dataset, dataset)
    assert 40 == new.length().result()
    dataset.append(dataset)
    assert 40 == dataset.length().result()


def test_dataset_slice(dataset):
    part = dataset[:8]
    assert part.length().result() == 8
    state = dataset[8]
    data = Dataset(dataset.context, atoms_list=[state])
    assert data.length().result() == 1


def test_dataset_from_xyz(context, tmp_path):
    data = generate_emt_cu_data(20)
    path_xyz = tmp_path / 'data.xyz'
    with open(path_xyz, 'w') as f:
        write_extxyz(f, data)
    dataset = Dataset.load(context, path_xyz)

    for i in range(20):
        assert np.allclose(
                data[i].get_positions(),
                dataset[i].result().get_positions(),
                )


def test_dataset_metric(context, dataset):
    errors = dataset.get_errors(intrinsic=True)
    assert errors.result().shape == (dataset.length().result(), 3)
    errors = np.mean(errors.result(), axis=0)
    assert np.all(errors > 0)
    with pytest.raises(AssertionError):
        errors = dataset.get_errors(intrinsic=True, atom_indices=[0, 1])
        errors.result()
    with pytest.raises(AssertionError):
        errors = dataset.get_errors(intrinsic=True, atom_indices=['C'])
        errors.result()
    errors = dataset.get_errors(intrinsic=True, elements=['H'], properties=['forces']) # H present
    errors.result()
    errors_rmse = dataset.get_errors(intrinsic=True, elements=['Cu'], properties=['forces']) # Cu present
    errors_mae  = dataset.get_errors(intrinsic=True, elements=['Cu'], properties=['forces'], metric='mae') # Cu present
    errors_max  = dataset.get_errors(intrinsic=True, elements=['Cu'], properties=['forces'], metric='max') # Cu present
    assert np.all(errors_rmse.result() > errors_mae.result())
    assert np.all(errors_max.result() > errors_rmse.result())

    with pytest.raises(AssertionError): # no atoms of interest
        errors = dataset.get_errors(intrinsic=True, elements=['O'], properties=['forces']) # Cu present
        errors.result()

    atoms = Atoms(numbers=30 * np.ones(10), positions=np.zeros((10, 3)), pbc=False)
    atoms.info['forces'] = np.random.uniform(-1, 1, size=(10, 3))
    dataset_ = Dataset(context, atoms_list=[atoms])
    merged = Dataset.merge(dataset, dataset_)
    errors = merged.get_errors(intrinsic=True, elements=['H'], properties=['forces']) # H present
    assert errors.result().shape[0] == merged.length().result() - 1


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
