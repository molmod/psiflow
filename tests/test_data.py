import os
import pytest
import numpy as np
from parsl.app.futures import DataFuture

from ase import Atoms
from ase.io import read, write
from ase.io.extxyz import write_extxyz

import psiflow
from psiflow.data import FlowAtoms, Dataset, NullState
from psiflow.utils import get_index_element_mask, is_reduced


def test_flow_atoms(context, dataset, tmp_path):
    atoms = dataset.get(index=0).result().copy()  # copy necessary with HTEX!
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
    atoms.cell[:] = np.array([[3, 1, 1], [1, 5, 0], [0, -1, 5]])
    assert not "energy" in atoms.info
    assert atoms.reference_status == False
    assert tuple(sorted(atoms.elements)) == ("Cu", "H")
    assert not is_reduced(atoms.cell)
    atoms.canonical_orientation()
    assert is_reduced(atoms.cell)


def test_dataset_empty(context, tmp_path):
    dataset = Dataset(atoms_list=[])
    assert dataset.length().result() == 0
    assert isinstance(dataset.data_future, DataFuture)
    path_xyz = tmp_path / "test.xyz"
    dataset.save(path_xyz)  # ensure the copy is executed before assert
    assert not os.path.isfile(path_xyz)
    psiflow.wait()
    assert os.path.isfile(path_xyz)
    with pytest.raises(ValueError):  # cannot save outside cwd
        dataset.save(path_xyz.parents[3] / "test.xyz")


def test_dataset_append(dataset):
    assert 20 == dataset.length().result()
    atoms_list = dataset.as_list().result()
    assert len(atoms_list) == 20
    assert type(atoms_list) == list
    assert type(atoms_list[0]) == FlowAtoms
    empty = Dataset([])  # use [] instead of None
    empty.append(dataset)
    assert 20 == empty.length().result()
    dataset.append(dataset)
    assert 40 == dataset.length().result()
    added = dataset + dataset
    assert added.length().result() == 80
    assert dataset.length().result() == 40  # may not changed
    dataset += dataset
    assert dataset.length().result() == 80  # may not changed

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


def test_dataset_from_xyz(context, tmp_path, dataset):
    path_xyz = tmp_path / "data.xyz"
    dataset.save(path_xyz)
    psiflow.wait()
    loaded = Dataset.load(path_xyz)
    data = read(path_xyz, index=":")

    for i in range(20):
        assert np.allclose(
            dataset[i].result().get_positions(),
            loaded[i].result().get_positions(),
        )
        assert np.allclose(
            dataset[i].result().get_positions(),
            data[i].get_positions(),
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
        errors = Dataset.get_errors(dataset, None, elements=["C"])
        errors.result()
    # should return empty array
    errors = Dataset.get_errors(dataset, None, elements=["Ne"], properties=["forces"])
    assert errors.result().shape == (0, 1)

    errors = Dataset.get_errors(
        dataset, None, elements=["H"], properties=["forces"]
    )  # H present
    errors.result()
    errors_rmse = Dataset.get_errors(
        dataset, None, elements=["Cu"], properties=["forces"]
    )  # Cu present
    errors_mae = Dataset.get_errors(
        dataset, None, elements=["Cu"], properties=["forces"], metric="mae"
    )  # Cu present
    errors_max = Dataset.get_errors(
        dataset, None, elements=["Cu"], properties=["forces"], metric="max"
    )  # Cu present
    assert np.all(errors_rmse.result() > errors_mae.result())
    assert np.all(errors_max.result() > errors_rmse.result())

    atoms = FlowAtoms(numbers=30 * np.ones(10), positions=np.zeros((10, 3)), pbc=False)
    atoms.info["forces"] = np.random.uniform(-1, 1, size=(10, 3))
    dataset.append(Dataset([atoms]))
    errors = Dataset.get_errors(
        dataset, None, elements=["H"], properties=["forces"]
    )  # H present
    assert errors.result().shape[0] == dataset.length().result() - 1


def test_index_element_mask():
    numbers = np.array([1, 1, 1, 6, 6, 8, 8, 8])
    elements = ["H"]
    indices = [0, 1, 4, 5]
    assert np.allclose(
        get_index_element_mask(numbers, elements, indices),
        np.array([True, True, False, False, False, False, False, False]),
    )
    elements = ["H", "O"]
    indices = [0, 1, 4, 5]
    assert np.allclose(
        get_index_element_mask(numbers, elements, indices),
        np.array([True, True, False, False, False, True, False, False]),
    )
    elements = ["H", "O"]
    indices = [3]
    assert np.allclose(
        get_index_element_mask(numbers, elements, indices),
        np.array([False] * len(numbers)),
    )
    elements = ["H", "O"]
    indices = None
    assert np.allclose(
        get_index_element_mask(numbers, elements, indices),
        np.array([True, True, True, False, False, True, True, True]),
    )
    elements = None
    indices = [0, 1, 2, 3, 5]
    assert np.allclose(
        get_index_element_mask(numbers, elements, indices),
        np.array([True, True, True, True, False, True, False, False]),
    )
    elements = ["Cl"]  # not present at all
    indices = None
    assert np.allclose(
        get_index_element_mask(numbers, elements, indices),
        np.array([False] * len(numbers)),
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
    assert "H" in dataset.elements().result()
    assert "Cu" in dataset.elements().result()
    assert len(dataset.elements().result()) == 2


def test_data_reset(context, dataset):
    dataset = dataset.reset()
    assert not "energy" in dataset[0].result().info


def test_nullstate(context):
    state = FlowAtoms(
        numbers=np.array([0]),
        positions=np.array([[0.0, 0, 0]]),
        pbc=False,
    )
    assert not id(state) == id(NullState)
    assert state == NullState


def test_data_split(context, dataset):
    train, valid = dataset.split(0.9)
    assert (
        train.length().result() + valid.length().result() == dataset.length().result()
    )


def test_identifier(context, dataset):
    data = dataset + Dataset([NullState, NullState, dataset[0].result()])
    identifier = data.assign_identifiers(0)
    assert identifier.result() == dataset.length().result() + 1
    assert identifier.result() == data.labeled().length().result()
    for i in range(data.length().result()):
        if not data[i].result() == NullState:
            assert data[i].result().info["identifier"] < identifier.result()
            assert data[i].result().reference_status
    data = data.reset()
    for i in range(data.length().result()):
        s = data[i].result()
        if not s == NullState:
            assert not "identifier" in s.info
    identifier = data.assign_identifiers(10)
    assert identifier.result() == 10  # none are labeled
    identifier = dataset.assign_identifiers(10)
    assert dataset.assign_identifiers().result() == 10 + dataset.length().result()
    for i in range(dataset.length().result()):
        s = dataset[i].result()
        if not s == NullState:
            assert s.info["identifier"] >= 10


def test_data_offset(context, dataset):
    atomic_energies = {
        "H": 34,
        "Cu": 12,
    }
    data = dataset.subtract_offset(**atomic_energies)
    data_ = data.add_offset(**atomic_energies)
    natoms = len(dataset[0].result())
    offset = (natoms - 1) * atomic_energies["Cu"] + atomic_energies["H"]
    for i in range(dataset.length().result()):
        assert np.allclose(
            data[i].result().info["energy"],
            dataset[i].result().info["energy"] - offset,
        )
        assert np.allclose(  # unchanged
            data[i].result().arrays["forces"],
            dataset[i].result().arrays["forces"],
        )
        assert np.allclose(
            data_[i].result().info["energy"],
            dataset[i].result().info["energy"],
        )
