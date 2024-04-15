import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.data import Dataset, Geometry, NullState
from psiflow.data.geometry import (
    _read_frames,
    _write_frames,
    check_equality,
    read_frames,
)
from psiflow.data.utils import get_index_element_mask


def test_geometry(tmp_path):
    geometry = Geometry.from_data(
        numbers=np.arange(1, 6),
        positions=np.random.uniform(0, 1, size=(5, 3)),
        cell=None,
    )
    assert not geometry.periodic

    _ = np.array(geometry.per_atom.positions)
    atoms = Atoms(
        numbers=np.arange(1, 6),
        positions=geometry.per_atom.positions,
        pbc=False,
    )
    geometry_ = Geometry.from_atoms(atoms)
    assert geometry == geometry_

    geometry = Geometry.from_data(
        numbers=np.arange(1, 6),
        positions=geometry_.per_atom.positions,
        cell=2 * np.eye(3),
    )
    assert geometry.periodic
    assert not geometry == geometry_
    address = geometry.per_atom.positions.ctypes.data
    address_ = geometry_.per_atom.positions.ctypes.data
    assert address != address_  # should copy
    assert np.allclose(
        geometry.per_atom.positions,
        geometry_.per_atom.positions,
    )
    geometry.align_axes()
    assert geometry.per_atom.positions.ctypes.data == address  # should not copy

    fake_data = []
    for i in range(10):
        atoms = Atoms(
            numbers=np.arange(i + 5),
            positions=np.random.uniform(-3, 3, size=(i + 5, 3)),
            pbc=False,
        )
        fake_data.append(atoms)
    fake_data[2].info["energy"] = 1.0
    fake_data[3].arrays["forces"] = np.random.uniform(-3, 3, size=(8, 3))
    fake_data[1].info["stdout"] = "abcd"
    fake_data[6].info["logprob"] = np.array([-23.1, 22.1])
    fake_data[7].cell = np.random.uniform(-3, 4, size=(3, 3))
    fake_data[7].pbc = True

    write(tmp_path / "test.xyz", fake_data)

    data = read_frames(inputs=[str(tmp_path / "test.xyz")]).result()
    assert data is not None  # should return when no output file is given
    assert len(data) == len(fake_data)
    for i in range(10):
        assert np.allclose(
            fake_data[i].get_positions(),
            data[i].per_atom.positions,
        )
        assert np.allclose(
            fake_data[i].numbers,
            data[i].per_atom.numbers,
        )
        assert data[i] == Geometry.from_atoms(fake_data[i])
    assert data[2].energy == 1.0
    assert np.allclose(
        data[3].per_atom.forces,
        fake_data[3].arrays["forces"],
    )
    assert data[1].stdout == "abcd"
    assert type(data[6].logprob) is np.ndarray
    assert np.allclose(
        data[6].logprob,
        fake_data[6].info["logprob"],
    )
    assert data[7].periodic
    assert np.allclose(
        data[7].cell,
        np.array(fake_data[7].cell),
    )
    assert np.allclose(
        data[6].cell,
        np.zeros((3, 3)),
    )
    geometry = read_frames(indices=[3], inputs=[str(tmp_path / "test.xyz")]).result()[0]
    assert geometry == data[3]
    assert not geometry == data[4]

    # check writing
    data = read_frames(
        inputs=[str(tmp_path / "test.xyz")],
        outputs=[File(str(tmp_path / "test_.xyz"))],
    ).result()
    assert data is None

    data = read_frames(inputs=[File(str(tmp_path / "test.xyz"))]).result()
    data_ = read_frames(inputs=[File(str(tmp_path / "test_.xyz"))]).result()

    for state, state_ in zip(data, data_):
        assert state == state_
        assert state.energy == state_.energy
        if state.stress is not None:
            assert np.allclose(
                state.stress,
                state_.stress,
            )
        assert state.delta == state_.delta
        assert state.phase == state_.phase
        if state.logprob is not None:
            assert np.allclose(state.logprob, state_.logprob)
        assert state.stdout == state_.stdout
        assert state.identifier == state.identifier


def test_readwrite_cycle(dataset, tmp_path):
    data = dataset[:4].geometries().result()
    data[2].order["test"] = 324
    _write_frames(*data, outputs=[str(tmp_path / "test.xyz")])
    loaded = Dataset(data)
    assert "test" in loaded[2].result().order

    states = _read_frames(inputs=[str(tmp_path / "test.xyz")])
    assert "test" in states[2].order


def test_dataset_empty(tmp_path):
    dataset = Dataset([])
    assert dataset.length().result() == 0
    assert isinstance(dataset.extxyz, DataFuture)
    path_xyz = tmp_path / "test.xyz"
    dataset.save(path_xyz)  # ensure the copy is executed before assert
    psiflow.wait()
    assert os.path.isfile(path_xyz)
    with pytest.raises(ValueError):  # cannot save outside cwd
        dataset.save(path_xyz.parents[3] / "test.xyz")


def test_dataset_append(dataset):
    l = dataset.length().result()  # noqa: E741
    geometries = dataset.geometries().result()
    assert len(geometries) == l
    assert type(geometries) is list
    assert type(geometries[0]) is Geometry
    for i in range(l):
        assert check_equality(geometries[i], dataset[i]).result()
    empty = Dataset([])  # use [] instead of None
    empty += dataset
    assert l == empty.length().result()
    # dataset.append(dataset)
    # assert 2 * l == dataset.length().result()
    added = dataset + dataset
    assert added.length().result() == 2 * l
    assert dataset.length().result() == l  # must not have changed

    # test canonical transformation
    transformed = dataset.align_axes()
    for i in range(dataset.length().result()):
        geometry = dataset[i].result()
        geometry.align_axes()
        assert np.allclose(
            geometry.per_atom.positions,
            transformed[i].result().per_atom.positions,
        )
        assert np.allclose(
            geometry.cell,
            transformed[i].result().cell,
        )


def test_dataset_slice(dataset):
    part = dataset[:8]
    assert part.length().result() == 8
    state = dataset[8]
    data = Dataset([state])
    assert data.length().result() == 1

    dataset_ = dataset.shuffle()
    equal = np.array([False] * dataset.length().result())
    for i in range(dataset.length().result()):
        pos0 = dataset_[i].result().per_atom.positions
        pos1 = dataset[i].result().per_atom.positions
        if pos0.shape == pos1.shape:
            equal[i] = np.allclose(pos0, pos1)
        else:
            equal[i] = False
    assert not np.all(equal)


def test_dataset_from_xyz(tmp_path, dataset):
    path_xyz = tmp_path / "data.xyz"
    dataset.save(path_xyz)
    psiflow.wait()
    _ = Dataset.load(path_xyz)
    atoms_list = read(path_xyz, index=":")

    for i in range(dataset.length().result()):
        assert np.allclose(
            dataset[i].result().per_atom.positions,
            atoms_list[i].get_positions(),
        )
        assert dataset[i].result().energy == atoms_list[i].info["energy"]


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


def test_dataset_gather(dataset):
    indices = [0, 3, 2, 6, 1]
    gathered = dataset[indices]
    assert gathered.length().result() == len(indices)
    for i, index in enumerate(indices):
        assert np.allclose(
            dataset[index].result().per_atom.positions,
            gathered[i].result().per_atom.positions,
        )


def test_data_elements(dataset):
    assert "H" in dataset.elements().result()
    assert "Cu" in dataset.elements().result()
    print(dataset.elements().result())
    assert len(dataset.elements().result()) == 2


def test_data_reset(dataset):
    dataset = dataset.reset()
    assert dataset[0].result().energy is None


def test_data_split(dataset):
    train, valid = dataset.split(0.9)
    assert (
        train.length().result() + valid.length().result() == dataset.length().result()
    )


def test_identifier(dataset):
    data = dataset + Dataset([NullState, NullState, dataset[0]])
    identifier = data.assign_identifiers(0)
    assert identifier.result() == dataset.length().result() + 1
    assert identifier.result() == data.not_null().length().result()
    for i in range(data.length().result()):
        if not data[i].result() == NullState:
            assert data[i].result().identifier < identifier.result()
            # assert data[i].result().reference_status
    data = data.clean()  # also removes identifier
    for i in range(data.length().result()):
        s = data[i].result()
        if not s == NullState:
            assert s.identifier is None
    identifier = data.assign_identifiers(10)
    assert identifier.result() == 10 + data.not_null().length().result()
    identifier = dataset.assign_identifiers(10)
    assert (
        dataset.assign_identifiers().result()
        == 10 + dataset.not_null().length().result()
    )
    for i in range(dataset.length().result()):
        s = dataset[i].result()
        if not s == NullState:
            assert s.identifier >= 10

    identifier = data.assign_identifiers()
    data = data.clean()
    assert (
        data.assign_identifiers(identifier).result()
        == identifier.result() + data.not_null().length().result()
    )


def test_data_offset(dataset):
    atomic_energies = {
        "H": 34,
        "Cu": 12,
    }
    data = dataset.subtract_offset(**atomic_energies)
    data_ = data.add_offset(**atomic_energies)
    for i in range(dataset.length().result()):
        natoms = len(dataset[i].result())
        offset = (natoms - 1) * atomic_energies["Cu"] + atomic_energies["H"]
        assert np.allclose(
            data[i].result().energy,
            dataset[i].result().energy - offset,
        )
        assert np.allclose(  # unchanged
            data[i].result().per_atom.forces,
            dataset[i].result().per_atom.forces,
        )
        assert np.allclose(
            data_[i].result().energy,
            dataset[i].result().energy,
        )


def test_data_extract(dataset):
    state = dataset[0].result()
    state.energy = None
    state.identifier = 0
    state.delta = 6

    data = dataset[:5] + dataset[-5:] + Dataset([state])
    energy, forces, identifier = data.get("energy", "forces", "identifier")
    energy = energy.result()
    forces = forces.result()
    identifier = identifier.result()
    for i, geometry in enumerate(data.geometries().result()):
        if geometry.energy:
            assert np.allclose(geometry.energy, energy[i])
        n = len(geometry)
        assert np.allclose(
            geometry.per_atom.forces,
            forces[i][:n, :],
        )
        if identifier[i].item() != -1:
            assert geometry.identifier == identifier[i]
        if i < 10:
            assert identifier[i] == -1
    assert np.isnan(np.mean(energy))
    assert np.isnan(np.mean(forces))
    psiflow.wait()
