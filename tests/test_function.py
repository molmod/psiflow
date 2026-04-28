import json

import numpy as np
from ase.units import kJ, mol  # type: ignore
from parsl.data_provider.files import File  # type: ignore

import psiflow
from psiflow.functions import (
    EinsteinCrystalFunction,
    HarmonicFunction,
    MACEFunction,
    PlumedFunction,
    function_from_json,
)
from psiflow.hamiltonians import (
    EinsteinCrystal,
    Harmonic,
    MixtureHamiltonian,
    PlumedHamiltonian,
    D3Hamiltonian,
    Zero,
)
from psiflow.utils._plumed import remove_comments_printflush, set_path_in_plumed
from psiflow.utils.apps import copy_app_future
from psiflow.utils.io import dump_json
from psiflow.utils.apps import get_attribute
from psiflow.serialization import CLS_KEY, deserialize_hook


def test_einstein_crystal(dataset):
    future_geom = dataset[0]
    test_dataset = dataset[:4].reset()
    geometries = test_dataset.geometries().result()

    function = EinsteinCrystalFunction(
        force_constant=1.0, centers=future_geom.result().per_atom.positions
    )
    hamiltonian = EinsteinCrystal.from_geometry(future_geom, force_constant=1.0)

    energy, forces, stress = function.compute(geometries).values()
    forces1, stress1, energy1 = hamiltonian.compute(
        test_dataset, "forces", "stress", "energy"
    )
    forces2 = hamiltonian.compute(test_dataset, "forces", batch_size=3)

    assert np.all(energy >= 0)
    assert energy[0] == 0
    assert geometries[0].energy is None
    assert np.allclose(  # forces point to centers
        forces,
        function.centers.reshape(1, -1, 3) - test_dataset.get("positions").result(),
    )
    assert np.allclose(energy1.result(), energy)
    assert np.allclose(forces1.result(), forces)
    assert np.allclose(forces1.result(), forces2.result())


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

    forces_ = einstein.compute(geometries, "forces").result()
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
    assert (
        plumed_input.strip()
        == """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV0: CV
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld FILE=/tmp/my_input
""".strip()
    )


def test_plumed_function(tmp_path, dataset, dataset_h2):
    data = dataset + dataset_h2
    geometries = data.geometries().result()
    plumed_str = """
D1: DISTANCE ATOMS=1,2 NOPBC
CV: BIASVALUE arg=D1
"""
    function = PlumedFunction(plumed_str)
    outputs = function.compute(geometries)

    f = 1 / (kJ / mol) * 10  # eV --> kJ/mol and nm --> A
    positions = data.get("positions").result()
    manual = np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=1)
    assert np.allclose(outputs["energy"] * f, manual)
    gradient = (positions[:, 0, :] - positions[:, 1, :]) / manual.reshape(-1, 1)
    assert np.allclose(outputs["forces"][:, 0, :] * f, gradient * (-1.0))

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

    energy, forces, stress = function.compute(geometries).values()
    energy_, forces_, stress_ = hamiltonian.compute(dataset)
    energy_manual = (volumes - 50) ** 2 * (kJ / mol) / 2
    assert np.allclose(energy, energy_manual)
    assert np.allclose(energy, energy_.result())
    assert np.allclose(stress, stress_.result())

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
""".format(
        path_hills
    )
    hamiltonian = PlumedHamiltonian(plumed_input, File(path_hills))
    energy = hamiltonian.compute(test_set, "energy")

    # compute bias energy manually
    positions = test_set.get("positions").result()
    vecs = positions[:, 0, :] - positions[:, 1, :]
    distance = np.linalg.norm(vecs, axis=1).reshape(-1, 1)
    sigma = 2 * np.ones((1, 2))
    height = np.array([70, 70]).reshape(1, -1) * (kJ / mol)  # unit consistency
    center = np.array([2.5, 2.6]).reshape(1, -1)
    energy_per_hill = height * np.exp((distance - center) ** 2 / (-2 * sigma**2))
    energy_ = np.sum(energy_per_hill, axis=1)
    assert np.allclose(energy.result(), energy_, atol=1e-3)

    # check that hills file didn't change
    with open(path_hills, "r") as f:
        assert f.read() == hills


def test_harmonic_function(dataset):
    test_set = dataset[:10]
    geometries = test_set.geometries().result()
    reference = geometries[0]
    hess = np.eye(3 * len(reference))

    function = HarmonicFunction(reference.per_atom.positions, hess, reference.energy)
    einstein = EinsteinCrystalFunction(1.0, reference.per_atom.positions)
    harmonic = Harmonic.from_geometry(dataset[0], hess)

    energy, forces, _ = function.compute(geometries).values()
    energy1, forces1, _ = einstein.compute(geometries).values()
    energy2, forces2, _ = harmonic.compute(test_set)

    assert Harmonic.outputs == ("energy", "forces", "stress")
    assert np.allclose(energy - reference.energy, energy1)
    assert np.allclose(forces, forces1)
    assert np.allclose(energy2.result() - reference.energy, energy1, atol=1e-5)
    assert np.allclose(forces2.result(), forces1)


def test_dispersion_function(dataset):
    hamiltonian = D3Hamiltonian(method="pbe", damping="d3bj")
    energy = hamiltonian.compute(dataset[-1], "energy").result()
    assert energy is not None
    assert energy < 0.0  # dispersion is attractive

    subset = dataset[:3]
    data = subset.evaluate(hamiltonian)
    energy, forces = hamiltonian.compute(subset, "energy", "forces")

    assert np.allclose(data.get("energy").result(), energy.result())
    assert len(forces.result().shape) == 3


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
    scaled_m = 0.5 * hamiltonian
    scaled_h1 = EinsteinCrystal.from_geometry(future, 0.5)
    assert len(scaled_m) == 1
    assert scaled_m.get_coefficient(hamiltonian) == 0.5
    assert scaled_m.get_coefficient(scaled_h1) is None
    scaled_h2 = EinsteinCrystal.from_geometry(future, 4.0)
    mixture = hamiltonian + scaled_h2
    assert len(mixture) == 2
    assert mixture == 0.9 * scaled_h2 + 0.1 * scaled_h2 + 1.0 * hamiltonian
    assert mixture.get_coefficients(mixture) == (1, 1)
    assert mixture.get_coefficients(hamiltonian + scaled_h1) is None

    energy_m = scaled_m.compute(test_set, "energy")
    energy_h1 = scaled_h1.compute(test_set, "energy")
    energy, forces, _ = hamiltonian.compute(test_set)
    energy_, forces_, _ = mixture.compute(test_set)
    assert np.allclose(energy_m.result(), energy_h1.result())
    assert np.allclose(energy_.result(), 5 * energy.result())
    assert np.allclose(forces_.result(), 5 * forces.result())

    energy, forces, stress = zero.compute(test_set)
    assert np.allclose(energy.result(), 0.0)
    geometries = [dataset[i].result() for i in [0, -1]]
    natoms = [len(geometry) for geometry in geometries]
    forces = zero.compute(geometries, "forces", batch_size=1).result()
    assert np.all(forces[0, : natoms[0]] == 0.0)
    assert np.all(forces[-1, : natoms[1]] == 0.0)


def test_subtract(dataset):
    einstein = EinsteinCrystal.from_geometry(dataset[0], force_constant=1.0)
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
    """.format(
        center=100, kappa=1 / (kJ / mol)
    )
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
    energy_e = einstein.compute(test_set, "energy")
    energy_e_ = einstein_.compute(test_set, "energy")
    energy_m = mixed.compute(test_set, "energy")
    energy_m_ = mixed_.compute(test_set, "energy")
    assert np.allclose(energy_e.result(), energy_e_.result())
    assert np.allclose(energy_m.result(), energy_m_.result())


def test_evaluate(dataset):
    hamiltonian = EinsteinCrystal.from_geometry(geometry=dataset[0], force_constant=1.0)
    data = dataset[:10].reset()
    evaluated = data.evaluate(hamiltonian, batch_size=None)
    evaluated_ = data.evaluate(hamiltonian, batch_size=2)
    energy = hamiltonian.compute(evaluated, "energy")

    geometries = evaluated.geometries().result()
    geometries_ = evaluated_.geometries().result()
    for geom, geom_, e in zip(geometries, geometries_, energy.result()):
        assert not np.all(np.isnan(geom.per_atom.forces))
        assert geom == geom_
        assert geom.energy == e


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
""".format(
        path_hills
    )
    hamiltonian = PlumedHamiltonian(plumed_input, File(path_hills))

    future = hamiltonian.serialize_function()
    future.result()  # ensure file exists
    function = function_from_json(future.filepath)

    energies = hamiltonian.compute(dataset, "energy")
    energies_ = function.compute(dataset.geometries().result())["energy"]
    assert np.allclose(energies.result(), energies_)
