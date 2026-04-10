import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

from psiflow.geometry import Geometry, MISSING
from psiflow.data import Dataset
from psiflow.data.file import _read_frames, _write_frames
from psiflow.data.utils import insert


def test_geometry(tmp_path):
    # TODO: custom per_atom field
    geometry = Geometry.from_data(
        numbers=np.arange(1, 6),
        positions=np.random.uniform(0, 1, size=(5, 3)),
        cell=None,
    )
    assert not geometry.periodic

    _ = np.array(geometry.per_atom.positions)
    atoms = Atoms(
        numbers=np.arange(1, 6), positions=geometry.per_atom.positions, pbc=False
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
    assert np.allclose(geometry.per_atom.positions, geometry_.per_atom.positions)
    geometry.align_axes()
    assert geometry.per_atom.positions.ctypes.data == address  # should not copy

    # create atoms dataset
    atoms = []
    for i in range(10):
        at = Atoms(
            numbers=np.arange(i + 5),
            positions=np.random.uniform(-3, 3, size=(i + 5, 3)),
            pbc=False,
        )
        atoms.append(at)
    atoms[2].info["energy"] = 1.0
    atoms[3].arrays["forces"] = np.random.uniform(-3, 3, size=(8, 3))
    atoms[1].info["stdout"] = "abcd"
    atoms[6].info["logprob"] = np.array([-23.1, 22.1])
    atoms[7].cell = np.random.uniform(-3, 4, size=(3, 3))
    atoms[7].pbc = True

    write(tmp_path / "atoms.xyz", atoms)

    # check reading from extxyz
    data = _read_frames(tmp_path / "atoms.xyz")
    assert data is not None  # should return when no output file is given
    assert len(data) == len(atoms)
    for at, geom in zip(atoms, data):
        assert geom == Geometry.from_atoms(at)
    assert data[2].energy == 1.0
    assert data[2].per_atom_energy == 1.0 / 7
    assert data[3].per_atom_energy is MISSING
    assert np.allclose(data[3].per_atom.forces, atoms[3].arrays["forces"])
    assert data[1].stdout == "abcd"
    assert type(data[6].logprob) is np.ndarray
    assert np.allclose(data[6].logprob, atoms[6].info["logprob"])
    assert data[7].periodic
    assert np.allclose(data[7].cell, atoms[7].cell.array)
    assert data[6].cell is None

    geometry = _read_frames(tmp_path / "atoms.xyz", indices=[3])[0]
    assert geometry == data[3]
    assert not geometry == data[4]

    # check writing to extxyz
    geoms = [Geometry.from_atoms(at) for at in atoms]
    _write_frames(geoms, outputs=[File(tmp_path / "geometries.xyz")])
    atoms_ = read(tmp_path / "geometries.xyz", ":")
    for at, at_ in zip(atoms, atoms_):
        assert np.allclose(at.cell, at_.cell)
        data = at.arrays | at.info | (at.calc.results if at.calc else {})
        data_ = at_.arrays | at_.info | (at_.calc.results if at_.calc else {})
        for k in data:
            if k == "cell":
                pass
            elif isinstance(data[k], np.ndarray):
                assert np.allclose(data[k], data_[k])
            else:
                assert data[k] == data_[k]

    data = _read_frames(tmp_path / "atoms.xyz")
    data_ = _read_frames(tmp_path / "geometries.xyz")
    for state, state_ in zip(data, data_):
        assert state == state_
        assert state.energy == state_.energy
        assert state.stress is MISSING or np.allclose(state.stress, state_.stress)
        attrs = state.attributes()
        attrs_ = state_.attributes()
        for val, val_ in zip(attrs.values(), attrs_.values()):
            if isinstance(val, np.ndarray):
                assert np.allclose(val, val_)
            else:
                assert val == val_

    state.save(tmp_path / "geo.xyz")
    assert state == Geometry.load(tmp_path / "geo.xyz")

    # default attributes have type (and shape) requirements
    with pytest.raises(TypeError):
        state.energy = None
    with pytest.raises(AssertionError):
        state.cell = [[1, 2, 3]] * 3


def test_readwrite_cycle(dataset, tmp_path):
    tmp_file = File(tmp_path / "test.xyz")

    data = dataset[:4].geometries().result()
    data[2].test = 324
    _write_frames(*data, outputs=[tmp_file])
    loaded = Dataset(data)
    assert hasattr(loaded[2].result(), "test")
    states = _read_frames(tmp_file)
    assert hasattr(states[2], "test")

    s = """3
energy=3.0 phase=c7eq Properties=species:S:1:pos:R:3:momenta:R:3:forces:R:3
C 0 1 2 3 4 5 6 7 8
O 1 2 3 4 5 6 7 8 9
F 2 3 4 5 6 7 8 9 10
"""
    geometry = Geometry.from_string(s)
    assert len(geometry) == 3
    assert geometry.energy == 3.0
    assert geometry.phase == "c7eq"
    assert not geometry.periodic
    assert not np.any(np.isnan(geometry.per_atom.forces))
    assert np.allclose(geometry.numbers, np.array([6, 8, 9]))
    assert np.allclose(
        geometry.per_atom.forces, np.array([[6, 7, 8], [7, 8, 9], [8, 9, 10]])
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
    assert geometry.energy is MISSING

    geom = Geometry.from_data(np.ones(2), np.zeros((2, 3)), cell=None)
    geom.stress = np.array([np.nan] * 9).reshape(3, 3)
    string = geom.to_string()
    assert "nan" in string  # the stress field
    geom_ = Geometry.from_string(string)
    assert np.isnan(geom_.stress).all()


def test_dataset(dataset, tmp_path):
    """Verifying basic dataset functionality"""
    l = dataset.length().result()  # noqa: E741
    geometries = dataset.geometries().result()
    assert len(geometries) == l
    assert type(geometries) is list
    assert type(geometries[0]) is Geometry
    for i in range(l):
        assert geometries[i] == dataset[i].result()

    path_xyz = tmp_path / "data.xyz"
    dataset.save(path_xyz).result()
    assert os.path.isfile(path_xyz)
    with pytest.raises(ValueError):  # cannot save outside cwd
        dataset.save(path_xyz.parents[3] / "test.xyz")

    dataset = Dataset([])
    assert dataset.length().result() == 0
    assert isinstance(dataset.extxyz, DataFuture)


def test_dataset_append(dataset):
    l = dataset.length().result()
    empty = Dataset([])  # use [] instead of None
    empty += dataset
    assert l == empty.length().result()
    added = dataset + dataset
    assert added.length().result() == 2 * l
    assert dataset.length().result() == l  # must not have changed


def test_dataset_align(dataset):
    # test canonical transformation
    transformed = dataset.align_axes()
    geoms = dataset.geometries().result()
    geoms_ = transformed.geometries().result()
    for geom, geom_ in zip(geoms, geoms_):
        geom.align_axes()
        assert np.allclose(geom.per_atom.positions, geom_.per_atom.positions)
        assert np.allclose(geom.cell, geom_.cell)


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

    indices = [0, 3, 2, 6, 1]
    gathered = dataset[indices]
    assert gathered.length().result() == len(indices)
    for i, index in enumerate(indices):
        assert np.allclose(
            dataset[index].result().per_atom.positions,
            gathered[i].result().per_atom.positions,
        )


def test_dataset_from_xyz(tmp_path, dataset):
    path_xyz = tmp_path / "data.xyz"
    dataset.save(path_xyz).result()
    atoms_list = read(path_xyz, index=":")
    geoms = dataset.geometries().result()
    for geom, at in zip(geoms, atoms_list):
        assert np.allclose(geom.per_atom.positions, at.get_positions())
        assert geom.energy == at.calc.results["energy"]


# def test_index_element_mask():
#     numbers = np.array([1, 1, 1, 6, 6, 8, 8, 8])
#     elements = ["H"]
#     indices = [0, 1, 4, 5]
#     assert np.allclose(
#         np.array([True, True, False, False, False, False, False, False]),
#         get_index_element_mask(numbers, indices, elements),
#     )
#     elements = ["H", "O"]
#     indices = [0, 1, 4, 5]
#     assert np.allclose(
#         np.array([True, True, False, False, False, True, False, False]),
#         get_index_element_mask(numbers, indices, elements),
#     )
#     elements = ["H", "O"]
#     indices = [3]
#     assert np.allclose(
#         get_index_element_mask(numbers, indices, elements),
#         np.array([False] * len(numbers)),
#     )
#     elements = ["H", "O"]
#     indices = None
#     assert np.allclose(
#         get_index_element_mask(numbers, indices, elements),
#         np.array([True, True, True, False, False, True, True, True]),
#     )
#     elements = None
#     indices = [0, 1, 2, 3, 5]
#     assert np.allclose(
#         get_index_element_mask(numbers, indices, elements),
#         np.array([True, True, True, True, False, True, False, False]),
#     )
#     elements = ["Cl"]  # not present at all
#     indices = None
#     assert np.allclose(
#         get_index_element_mask(numbers, indices, elements),
#         np.array([False] * len(numbers)),
#     )


def test_data_elements(dataset):
    elements = dataset.elements().result()
    assert "H" in elements
    assert "Cu" in elements
    assert len(elements) == 2


def test_data_reset(dataset):
    dataset = dataset.reset()
    geometries = dataset.geometries().result()
    for geom in geometries:
        assert geom.energy is MISSING
        assert geom.per_atom.forces is MISSING


def test_data_split(dataset):
    train, valid = dataset.split(0.9)
    train_geoms = train.geometries().result()
    valid_geoms = valid.geometries().result()
    assert len(train_geoms) + len(valid_geoms) == dataset.length().result()
    for geom_t in train_geoms:
        for geom_v in valid_geoms:
            assert geom_t != geom_v


def test_identifier(dataset, dataset_h2):
    data = dataset + dataset_h2
    identifier = data.assign_identifiers(0)
    assert identifier.result() == data.length().result()

    data = data.clean()  # removes identifier
    geoms = data.geometries().result()
    for geom in geoms:
        assert not hasattr(geom, "identifier")

    identifier = dataset.assign_identifiers(5).result()
    geoms = dataset.geometries().result()
    assert identifier == len(geoms) + 5
    for geom in geoms:
        assert geom.identifier < identifier

    # set initial value
    geoms = dataset_h2.geometries().result()
    geoms[10].identifier = 42

    # count from 0
    data = Dataset(geoms)
    identifier = data.assign_identifiers(0).result()
    geoms_ = data.geometries().result()
    assert identifier == len(geoms) - 1  # skips over one geometry
    assert geoms_[0].identifier == 0
    assert geoms_[10].identifier == 42
    assert geoms_[-1].identifier == identifier - 1

    # count from highest
    data = Dataset(geoms)
    identifier = data.assign_identifiers().result()
    geoms_ = data.geometries().result()
    assert identifier == len(geoms) + 42  # skips over one geometry
    assert geoms_[0].identifier == 43
    assert geoms_[10].identifier == 42
    assert geoms_[-1].identifier == identifier - 1


def test_data_offset(dataset):
    atomic_energies = {"H": 34, "Cu": 12}
    data0 = dataset.subtract_offset(**atomic_energies)
    data1 = data0.add_offset(**atomic_energies)

    geoms = dataset.geometries().result()
    geoms0 = data0.geometries().result()
    geoms1 = data1.geometries().result()

    for g, g0, g1 in zip(geoms, geoms0, geoms1):
        offset = (len(g) - 1) * atomic_energies["Cu"] + atomic_energies["H"]
        assert np.allclose(g0.energy, g.energy - offset)
        assert np.allclose(g0.per_atom.forces, g.per_atom.forces)
        assert np.allclose(g1.energy, g.energy)


def test_data_extract(dataset):
    state = dataset[0].result()
    state.reset()
    state.identifier = 0
    state.delta = 6

    # check Dataset.get
    data = dataset[:5] + dataset[-5:] + Dataset([state])
    energy, forces, identifier = data.get("energy", "forces", "identifier")
    energy = energy.result()
    forces = forces.result()
    identifier = identifier.result()
    assert energy[-1] is MISSING and forces[-1] is MISSING
    assert identifier[-1] == 0
    for i, geometry in enumerate(data.geometries().result()):
        if geometry.energy:
            assert np.allclose(geometry.energy, energy[i])
        try:
            assert np.allclose(geometry.per_atom.forces, forces[i])
        except TypeError:
            assert geometry.per_atom.forces is MISSING and forces[i] is MISSING
        if hasattr(geometry, "identifier"):
            assert geometry.identifier == identifier[i]

    # check extract-insert round-trip
    geoms = data.clean().geometries().result()
    data_dict = {"energy": energy, "forces": forces, "identifier": identifier}
    insert(geoms, data_dict)
    for geom, geom_ in zip(geoms, data.geometries().result()):
        e, e_ = geom.energy, geom_.energy
        f, f_ = geom.per_atom.forces, geom_.per_atom.forces
        assert e == e_
        if f is MISSING:
            assert f_ is MISSING
        else:
            assert np.allclose(f, f_)
        if hasattr(geom, "identifier"):
            assert geom.identifier == geom_.identifier

    # check Dataset.get_per_atom
    geoms = data.geometries().result()
    forces = [geom.per_atom.forces for geom in geoms]
    numbers = [geom.numbers for geom in geoms]

    forces_ = data.get_per_atom("forces", elements=["Cu"])[0].result()
    for n, f, f_ in zip(numbers, forces, forces_):
        # filter on Cu
        if f is MISSING:
            assert f_ is MISSING
        else:
            assert np.allclose(f[n == 29], f_)

    ids = [0, 2]
    forces_ = data.get_per_atom("forces", atom_indices=ids)[0].result()
    for f, f_ in zip(forces, forces_):
        # filter on ids
        if f is MISSING:
            assert f_ is MISSING
        else:
            assert np.allclose(f[ids], f_)

    # check custom attributes
    s = Geometry.from_string(
        """
3
distance=2.3 energy=4
O 0 0 0
H 1 1 1
H -1 1 1
""",
    )
    states = [s.copy() for _ in range(5)]
    states[1].order = {"answer": 42}
    states[2].some = 4.2
    states[3].distance = 2.2
    data = Dataset(states)
    some, distance, energy, order = data.get("some", "distance", "energy", "order")
    some = some.result()
    distance = distance.result()
    energy = energy.result()
    order = order.result()
    assert some == [MISSING, MISSING, 4.2, MISSING, MISSING]
    assert distance == [2.3, 2.3, 2.3, 2.2, 2.3]
    assert energy == [4, 4, 4, 4, 4]
    assert order == [MISSING, {"answer": 42}, MISSING, MISSING, MISSING]


def test_filter(dataset, dataset_h2):
    geom = dataset[0].result()
    geom.flagged = True
    geom.reset()
    data = dataset + dataset_h2 + Dataset([geom])

    data0 = data.shuffle()
    data1 = data.filter("cell")
    data2 = data.filter("energy")
    data3 = data.filter("forces")
    data4 = data.filter("flagged")

    l = dataset.length().result()
    l_ = dataset_h2.length().result()
    l0 = data0.length().result()
    l1 = data1.length().result()
    l2 = data2.length().result()
    l3 = data3.length().result()
    l4 = data4.length().result()

    assert l0 == l + l_ + 1  # all geometries
    assert l0 == l1  # cell always defined
    assert l2 == l and l3 == l
    assert l4 == 1
