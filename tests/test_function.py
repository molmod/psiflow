import json

import numpy as np
from ase.units import kJ, mol  # type: ignore
from parsl.data_provider.files import File  # type: ignore

import psiflow
from psiflow.geometry import MISSING
from psiflow.functions import (
    EinsteinCrystalFunction,
    HarmonicFunction,
    MACEFunction,
    PlumedFunction,
    DispersionFunction,
    ZeroFunction,
    function_from_json,
)
from psiflow.hamiltonians import (
    EinsteinCrystal,
    Harmonic,
    MixtureHamiltonian,
    PlumedHamiltonian,
    D3Hamiltonian,
    MACEHamiltonian,
    Zero,
)
from psiflow.utils._plumed import remove_comments_printflush, set_path_in_plumed
from psiflow.utils.apps import copy_app_future
from psiflow.utils.io import dump_json
from psiflow.utils.apps import get_attribute
from psiflow.serialization import CLS_KEY, deserialize_hook
from psiflow.compute import compare_results


def test_einstein_crystal(dataset):
    future_geom = dataset[0]
    geom = future_geom.result()
    test_dataset = dataset[:4].reset()
    test_geoms = test_dataset.geometries().result()

    function = EinsteinCrystalFunction(
        force_constant=1.0, centers=geom.per_atom.positions, volume=geom.volume
    )
    hamiltonian = EinsteinCrystal.from_geometry(future_geom, force_constant=1.0)

    out0 = function.compute(test_geoms)
    assert all(e >= 0 for e in out0["energy"])
    assert out0["energy"][0] == 0
    assert test_geoms[0].energy is MISSING
    positions = test_dataset.get("positions")[0].result()
    centers = function.centers.reshape(1, -1, 3)
    for f, p in zip(out0["forces"], positions):
        assert np.allclose(f, centers - p)  # forces point to centers

    out1 = hamiltonian.compute(test_dataset)
    out2 = hamiltonian.compute(test_dataset, batch_size=3)

    error1 = compare_results(out1, **out0)
    error2 = compare_results(out1, out2)

    for k, rmse in error1.result().items():
        assert np.isclose(rmse, 0.0)
    for k, rmse in error2.result().items():
        assert np.isclose(rmse, 0.0)


def test_einstein_force(dataset):
    reference = dataset[0].result()
    force_constant = 5
    delta = 0.1
    einstein = EinsteinCrystal.from_geometry(reference, force_constant)

    geometries, forces = [], []
    for i in range(len(reference)):
        for j in range(3):  # x, y, z
            for sign in [+1, -1]:
                geometry = reference.copy()
                geometry.per_atom.positions[i, j] += sign * delta
                geometries.append(geometry)
                forces_ = np.zeros_like(geometry.per_atom.positions)
                forces_[i, j] = (-1.0) * sign * force_constant * delta
                forces.append(forces_)

    # per atom arrays are concatenated - not stacked
    forces_ = einstein.compute(geometries).result().forces
    forces_ = forces_.reshape(-1, len(reference), 3)
    assert np.allclose(forces, forces_)


def test_get_filename_hills():
    plumed_input = """
#METAD COMMENT TO BE REMOVED
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV0: CV #lkasdjf
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld #fjalsdkfj
PRINT ARG=CV,metad.bias STRIDE=10 FILE=COLVAR
FLUSH STRIDE=10
"""
    plumed_input = remove_comments_printflush(plumed_input)
    plumed_input = set_path_in_plumed(plumed_input, "METAD", "/tmp/my_input")
    plumed_input = set_path_in_plumed(plumed_input, "METADD", "/tmp/my_input")
    assert plumed_input.strip() == """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV0: CV
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld FILE=/tmp/my_input
""".strip()


def test_plumed_function(tmp_path, dataset, dataset_h2):
    data = dataset + dataset_h2
    geometries = data.geometries().result()
    f = 1 / (kJ / mol) * 10  # eV --> kJ/mol and nm --> A
    plumed_str = """
D1: DISTANCE ATOMS=1,2 NOPBC
CV: BIASVALUE arg=D1
"""
    function = PlumedFunction(plumed_str)
    outputs = function.compute(geometries)

    [positions] = data.get("positions")
    manual = np.array([np.linalg.norm(p[0] - p[1]) for p in positions.result()])
    gradient = [(p[0] - p[1]) / d for p, d in zip(positions.result(), manual)]
    assert np.allclose(np.array(outputs["energy"]) * f, manual)
    for grad, force in zip(gradient, outputs["forces"]):
        # only compare first force component
        assert np.allclose(force[0] * f, grad * (-1.0))

    # test volume bias
    geometries = dataset.geometries().result()
    volumes = np.array([g.volume for g in geometries])
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=50 KAPPA=1
"""
    function = PlumedFunction(plumed_input)
    hamiltonian = PlumedHamiltonian(plumed_input)

    data = function.compute(geometries)
    out = hamiltonian.compute(dataset).result()
    energy_manual = (volumes - 50) ** 2 * (kJ / mol) / 2
    assert np.allclose(data["energy"], energy_manual)
    assert np.allclose(data["energy"], out.energy)
    assert np.allclose(data["stress"], out.stress)

    # use external grid as bias, check that file is read
    test_set = dataset[:10]
    hills = """#! FIELDS time CV sigma_CV height biasf
#! SET multivariate false
#! SET kerneltype gaussian
     1.00000     2.5     2.0     70  0
     2.00000     2.6     2.0     70  0
"""
    path_hills = tmp_path / "hills"
    with open(path_hills, "w") as f:
        f.write(hills)
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=1 SIGMA=3 HEIGHT=342 FILE={}
""".format(path_hills)
    hamiltonian = PlumedHamiltonian(plumed_input, File(path_hills))
    energy = hamiltonian.compute(test_set).result().energy

    # compute bias energy manually
    sigma = 2 * np.ones((1, 2))
    height = np.array([70, 70]).reshape(1, -1) * (kJ / mol)  # unit consistency
    center = np.array([2.5, 2.6]).reshape(1, -1)
    positions = test_set.get("positions")[0].result()
    dists = np.array([np.linalg.norm(p[0] - p[1]) for p in positions]).reshape(-1, 1)
    energy_per_hill = height * np.exp((dists - center) ** 2 / (-2 * sigma**2))
    energy_ = np.sum(energy_per_hill, axis=1)
    assert np.allclose(energy, energy_, atol=1e-3)

    # check that hills file didn't change
    with open(path_hills, "r") as f:
        assert f.read() == hills


def test_harmonic_function(dataset):
    test_set = dataset[:5]
    geometries = test_set.geometries().result()
    reference = geometries[0]
    hess = np.eye(3 * len(reference))

    function = HarmonicFunction(reference.per_atom.positions, hess, reference.energy)
    einstein = EinsteinCrystalFunction(1.0, reference.per_atom.positions)
    harmonic = Harmonic.from_geometry(dataset[0], hess)
    assert Harmonic.outputs == ("energy", "forces", "stress")

    out0 = function.compute(geometries)
    out1 = einstein.compute(geometries)
    result = harmonic.compute(test_set)
    err0 = compare_results(result, **out0).result()
    err1 = compare_results(result, **out1).result()

    assert np.allclose(np.array(out0["energy"]) - reference.energy, out1["energy"])
    assert np.isclose(err0["energy"], 0)
    assert np.isclose(err1["energy"], reference.energy)
    assert np.allclose(out0["forces"], out1["forces"])
    assert np.isclose(err0["forces"], 0)
    assert np.isclose(err1["forces"], 0)


def test_dispersion_function(dataset):
    test_set = dataset[:3]
    function = DispersionFunction(method="pbe", damping="d3bj")
    hamiltonian = D3Hamiltonian(method="pbe", damping="d3bj")

    out = function.compute(test_set.geometries().result())
    result = hamiltonian.compute(test_set)
    error = compare_results(result, **out)
    for k, rmse in error.result().items():
        assert np.isclose(rmse, 0.0)
    assert np.all(np.array(out["energy"]) < 0.0)  # dispersion is attractive


def test_zero(dataset):
    test_set = dataset[:10]
    function = ZeroFunction()
    hamiltonian = Zero()

    out = function.compute(test_set.geometries().result())
    result = hamiltonian.compute(test_set).result()
    error = compare_results(result, **out)
    for k, rmse in error.result().items():
        assert np.isclose(rmse, 0.0)
    assert np.allclose(result.energy, 0.0)
    assert np.allclose(result.forces, 0.0)


def test_mace(dataset, mace_foundation):
    test_set = dataset[:3]
    function = MACEFunction(mace_foundation, 1, "cpu", "float32")
    hamiltonian = MACEHamiltonian(mace_foundation)

    out = function.compute(test_set.geometries().result())
    result = hamiltonian.compute(test_set).result()
    error = compare_results(result, **out)
    for k, rmse in error.result().items():
        assert np.isclose(rmse, 0.0, atol=1e-6)  # single precision


def test_hamiltonian_arithmetic(dataset):
    test_set = dataset[:10]
    future = dataset[0]
    geom = future.result()
    centers = get_attribute(future, "per_atom", "positions")
    volume = get_attribute(future, "volume")

    hamiltonian = EinsteinCrystal.from_geometry(geom, force_constant=1.0)
    assert hamiltonian == hamiltonian
    hamiltonian1 = EinsteinCrystal.from_geometry(geom, force_constant=1.0)
    assert hamiltonian == hamiltonian1
    hamiltonian1 = EinsteinCrystal.from_geometry(geom, force_constant=1.1)
    assert hamiltonian != hamiltonian1  # different constants
    hamiltonian1 = EinsteinCrystal(1.0, centers, volume)
    assert hamiltonian != hamiltonian1  # unknown future
    hamiltonian2 = EinsteinCrystal(1.0, centers, volume)
    assert hamiltonian1 == hamiltonian2  # same unknown future
    assert hamiltonian != PlumedHamiltonian(plumed_input="")

    # consider linear combinations
    zero = Zero()
    assert hamiltonian == hamiltonian + zero
    assert 2 * hamiltonian + zero == 2 * hamiltonian
    scaled_h = 0.5 * hamiltonian
    scaled_h1 = EinsteinCrystal.from_geometry(future, 0.5)
    assert len(scaled_h) == 1
    assert scaled_h.get_coefficient(hamiltonian) == 0.5
    assert scaled_h.get_coefficient(scaled_h1) is None
    scaled_h2 = EinsteinCrystal.from_geometry(future, 4.0)
    mixture = hamiltonian + scaled_h2
    assert len(mixture) == 2
    assert mixture == 0.9 * scaled_h2 + 0.1 * scaled_h2 + 1.0 * hamiltonian
    assert mixture.get_coefficients(mixture) == (1, 1)
    assert mixture.get_coefficients(hamiltonian + scaled_h1) is None

    result_h = scaled_h.compute(test_set)
    result_h1 = scaled_h1.compute(test_set)
    error = compare_results(result_h, result_h1)
    out = hamiltonian.compute(test_set)
    out_ = mixture.compute(test_set)
    error, out, out_ = error.result(), out.result(), out_.result()

    for k, v in error.items():
        assert np.isclose(v, 0)
    assert np.allclose(out.energy * 5, out_.energy)
    assert np.allclose(out.forces * 5, out_.forces)

    # check subtraction
    einstein = EinsteinCrystal.from_geometry(future, force_constant=1.0)
    h = einstein - einstein
    assert isinstance(h, MixtureHamiltonian)
    assert np.allclose(h.coefficients, 0.0)


def test_hamiltonian_serialize(dataset):
    data = json.loads(psiflow.serialize(Zero()).result())
    assert data[CLS_KEY] == "Zero"
    zero = psiflow.deserialize(json.dumps(data)).result()
    assert isinstance(zero, Zero)

    plumed_input = """
    UNITS LENGTH=A ENERGY=kj/mol TIME=fs
    CV: VOLUME
    RESTRAINT ARG=CV AT={center} KAPPA={kappa}
    """.format(center=100, kappa=1 / (kJ / mol))
    plumed = PlumedHamiltonian(plumed_input)
    einstein = EinsteinCrystal.from_geometry(dataset[0], force_constant=1.0)

    data = json.loads(psiflow.serialize(einstein).result())
    assert data[CLS_KEY] == "EinsteinCrystal"
    assert all((k in data) for k in ("centers", "volume"))
    einstein_ = psiflow.deserialize(json.dumps(data)).result()

    mixed = 0.1 * einstein + 0.9 * plumed
    data = json.loads(psiflow.serialize(mixed).result())
    assert data[CLS_KEY] == "MixtureHamiltonian"
    assert "hamiltonians" in data
    assert "coefficients" in data
    mixed_ = psiflow.deserialize(json.dumps(data)).result()
    assert mixed.coefficients == mixed_.coefficients
    for h, h_ in zip(mixed.hamiltonians, mixed_.hamiltonians):
        if isinstance(h, EinsteinCrystal):
            assert h.force_constant == h_.force_constant
            assert h.volume.result() == h_.volume
            assert np.allclose(h.centers.result(), h_.centers)
        else:
            assert h == h_

    test_set = dataset[:10]
    out1 = einstein.compute(test_set)
    out1_ = einstein_.compute(test_set)
    out2 = mixed.compute(test_set)
    out2_ = mixed_.compute(test_set)
    assert np.allclose(out1.result().energy, out1_.result().energy)
    assert np.allclose(out2.result().energy, out2_.result().energy)


def test_json_dump():
    data = {
        "a": np.ones((3, 3, 3, 2)),
        "b": [1, 2, 3],
        "c": (1, 2, 4),
        "d": "asdf",
        "e": copy_app_future(False),
    }
    data_future = dump_json(
        **data, outputs=[psiflow.context().new_file("bla_", ".json")]
    ).outputs[0]
    data_future.result()
    with open(data_future.filepath, "r") as f:
        data_ = json.loads(f.read(), object_hook=deserialize_hook)

    new_a = np.array(data_["a"])
    assert len(new_a.shape) == 4
    assert np.allclose(data["a"], new_a)
    assert data_["e"] is False
    assert type(data_["b"]) is list
    assert type(data_["c"]) is list


def test_function_from_json(tmp_path, dataset):
    hills = """#! FIELDS time CV sigma_CV height biasf
#! SET multivariate false
#! SET kerneltype gaussian
     1.00000     2.5     2.0     70  0
     2.00000     2.6     2.0     70  0
"""
    path_hills = tmp_path / "hills"
    with open(path_hills, "w") as f:
        f.write(hills)
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=1 SIGMA=3 HEIGHT=342 FILE={}
""".format(path_hills)
    hamiltonian = PlumedHamiltonian(plumed_input, File(path_hills))

    future = hamiltonian.serialize_function()
    future.result()  # ensure file exists
    function = function_from_json(future.filepath)

    energies = hamiltonian.compute(dataset).result().energy
    energies_ = function.compute(dataset.geometries().result())["energy"]
    assert np.allclose(energies, energies_)


def test_evaluate(dataset):
    test_set = dataset[:10].reset()
    hamiltonian = EinsteinCrystal.from_geometry(dataset[0], force_constant=1.0)

    out0 = hamiltonian.evaluate(test_set)
    out1 = hamiltonian.evaluate(test_set, batch_size=None)
    result = hamiltonian.compute(test_set)

    geoms0 = out0.geometries().result()
    geoms1 = out1.geometries().result()
    data = result.result().to_dict()

    for i in range(len(geoms0)):
        geom0, geom1 = geoms0[i], geoms1[i]
        assert np.isclose(geom0.energy, geom1.energy)
        assert np.isclose(geom0.energy, data["energy"][i])
        assert np.allclose(geom0.per_atom.forces, geom1.per_atom.forces)
        assert np.allclose(geom0.per_atom.forces, data["forces"][i])
        assert np.allclose(geom0.stress, geom1.stress)
        assert np.allclose(geom0.stress, data["stress"][i])

