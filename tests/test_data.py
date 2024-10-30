import copy
import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.data import Dataset, compute_rmse
from psiflow.data.utils import (
    _read_frames,
    _write_frames,
    get_index_element_mask,
    read_frames,
)
from psiflow.geometry import Geometry, NullState, check_equality


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
    assert data[2].per_atom_energy == 1.0 / 7
    assert data[3].per_atom_energy is None
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

    state.save(tmp_path / "geo.xyz")
    state.save(str(tmp_path / "geo.xyz"))
    assert state == Geometry.load(tmp_path / "geo.xyz")


def test_readwrite_cycle(dataset, tmp_path):
    data = dataset[:4].geometries().result()
    data[2].order["test"] = 324
    _write_frames(*data, outputs=[str(tmp_path / "test.xyz")])
    loaded = Dataset(data)
    assert "test" in loaded[2].result().order

    states = _read_frames(inputs=[str(tmp_path / "test.xyz")])
    assert "test" in states[2].order

    s = """3
energy=3.0 phase=c7eq Properties=species:S:1:pos:R:3:momenta:R:3:forces:R:3
C 0 1 2 3 4 5 6 7 8
O 1 2 3 4 5 6 7 8 9
F 2 3 4 5 6 7 8 9 10
"""
    geometry = Geometry.from_string(s, natoms=None)
    assert len(geometry) == 3
    assert geometry.energy == 3.0
    assert geometry.phase == "c7eq"
    assert not geometry.periodic
    assert np.all(np.logical_not(np.isnan(geometry.per_atom.forces)))
    assert np.allclose(
        geometry.per_atom.numbers,
        np.array([6, 8, 9]),
    )
    assert np.allclose(
        geometry.per_atom.forces,
        np.array([[6,7,8], [7,8,9], [8,9,10]]),
    )
    s = """7

O       0.269073490000000      0.952731530000000      0.639899630000000
C       0.379007320000000     -0.130111590000000      0.085677050000000
C      -0.611025150000000     -0.821472640000000     -0.827555540000000
H       1.276678450000000     -0.822788000000000      0.432385510000000
H      -0.364845780000000     -1.985677100000000     -0.791238000000000
H      -1.419992260000000     -0.472979080000000     -0.156752270000000
H      -0.764765500000000     -0.536376480000000     -1.844856190000000
"""
    geometry = Geometry.from_string(s)
    assert not geometry.periodic
    assert len(geometry) == 7
    assert geometry.energy is None

    geometry = Geometry.from_data(np.ones(2), np.zeros((2, 3)), cell=None)
    geometry.stress = np.array([np.nan] * 9).reshape(3, 3)
    assert "nan" not in geometry.to_string()


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

    geometries = dataset[:10].geometries()
    subset = Dataset(geometries)
    assert subset.length().result() == 10
    for i, geometry in enumerate(subset.geometries().result()):
        assert geometry == dataset[i].result()


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
        assert dataset[i].result().energy == atoms_list[i].calc.results["energy"]


def test_index_element_mask():
    numbers = np.array([1, 1, 1, 6, 6, 8, 8, 8])
    elements = ["H"]
    indices = [0, 1, 4, 5]
    assert np.allclose(
        np.array([True, True, False, False, False, False, False, False]),
        get_index_element_mask(numbers, indices, elements),
    )
    elements = ["H", "O"]
    indices = [0, 1, 4, 5]
    assert np.allclose(
        np.array([True, True, False, False, False, True, False, False]),
        get_index_element_mask(numbers, indices, elements),
    )
    elements = ["H", "O"]
    indices = [3]
    assert np.allclose(
        get_index_element_mask(numbers, indices, elements),
        np.array([False] * len(numbers)),
    )
    elements = ["H", "O"]
    indices = None
    assert np.allclose(
        get_index_element_mask(numbers, indices, elements),
        np.array([True, True, True, False, False, True, True, True]),
    )
    elements = None
    indices = [0, 1, 2, 3, 5]
    assert np.allclose(
        get_index_element_mask(numbers, indices, elements),
        np.array([True, True, True, True, False, True, False, False]),
    )
    elements = ["Cl"]  # not present at all
    indices = None
    assert np.allclose(
        get_index_element_mask(numbers, indices, elements),
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

    data = dataset[:2] + Dataset([NullState]) + dataset[3:5]
    forces = data.get("forces", elements=["Cu"])
    reference = np.zeros((5, 4, 3))
    reference[2, :] = np.nan  # ensure nan is in same place
    reference[:, 0] = np.nan  # ensure nan is in same place
    value = compute_rmse(forces, reference)

    # last three atoms are Cu
    forces = np.zeros((5, 4, 3))
    for i in range(5):
        forces[i, :] = data[i].result().per_atom.forces
    forces[:, 0] = np.nan
    assert np.allclose(
        value.result(),
        compute_rmse(forces, forces * np.zeros_like(forces)).result(),
    )
    unreduced = compute_rmse(
        forces, forces * np.zeros_like(forces), reduce=False
    ).result()
    assert len(unreduced) == 5
    unreduced_ = unreduced[np.array([0, 1, 3, 4], dtype=int)]
    assert np.allclose(
        np.sqrt(np.mean(np.square(unreduced_))),
        value.result(),
    )

    # try weirdly specific indices
    forces = np.zeros((5, 4, 3))
    for i in range(5):
        g = data[i].result()
        if len(g) >= 4:
            forces[i, 3] = data[i].result().per_atom.forces[3]
            forces[i, 1] = data[i].result().per_atom.forces[1]
        else:
            forces[i, :] = np.nan
    forces_ = data.get("forces", atom_indices=[3, 1]).result()
    mask = np.invert(np.isnan(forces_))
    assert np.allclose(
        forces[mask],
        forces_[mask],
    )

    # check order parameters
    s = Geometry.from_string(
        """
3
order_distance=2.3 energy=4
O 0 0 0
H 1 1 1
H -1 1 1
""",
    )
    states = [copy.deepcopy(s) for _ in range(5)]

    states[1].order = {}
    states[2].order = {"some": 4.2}
    states[3].order["distance"] = 2.2
    data = Dataset(states)
    some, distance = data.get("some", "distance")
    some = some.result()
    distance = distance.result()
    assert np.isnan(some[1])
    assert np.allclose(some[2], 4.2)
    assert np.allclose(distance[0], 2.3)
    assert np.allclose(distance[3], 2.2)
    assert np.isnan(distance[2])


def test_filter(dataset, dataset_h2):
    data = dataset + dataset_h2 + Dataset([NullState])
    data = data.shuffle()
    assert data.filter("cell").length().result() == dataset.length().result()
    assert data.filter("energy").length().result() == dataset.length().result()
    assert data.filter("forces").length().result() == dataset.length().result()
