import os
import pytest
import numpy as np
from parsl.app.futures import DataFuture

from ase.io.extxyz import write_extxyz

from autolearn import Dataset

from common import context, generate_emt_cu_data


@pytest.fixture
def dataset(context, tmp_path):
    data = generate_emt_cu_data(20)
    return Dataset(context, data)


def test_dataset_empty(context, tmp_path):
    dataset = Dataset(context, atoms_list=[])
    assert dataset.length().result() == 0
    assert isinstance(dataset.data, DataFuture)
    path_xyz = tmp_path / 'test.xyz'
    dataset.save(path_xyz).result() # ensure the copy is executed before assert
    assert os.path.isfile(path_xyz)


def test_dataset_add(dataset):
    assert 20 == dataset.length().result()
    new = Dataset.merge(dataset, dataset)
    assert 40 == new.length().result()


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
    dataset = Dataset.from_xyz(context, path_xyz)

    for i in range(20):
        assert np.allclose(
                data[i].get_positions(),
                dataset[i].result().get_positions(),
                )
