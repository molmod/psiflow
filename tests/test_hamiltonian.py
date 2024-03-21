import json

import numpy as np
from ase.units import kJ, mol
from parsl.data_provider.files import File

import psiflow
from psiflow.hamiltonians import EinsteinCrystal, PlumedHamiltonian, deserialize
from psiflow.hamiltonians._plumed import evaluate_plumed
from psiflow.hamiltonians.hamiltonian import Zero
from psiflow.utils import copy_app_future, copy_data_future, dump_json


def test_einstein(context, dataset):
    hamiltonian = EinsteinCrystal(dataset[0], force_constant=1)
    evaluated = hamiltonian.evaluate(dataset[:10])
    assert evaluated[0].result().info["energy"] == 0.0
    for i in range(1, 10):
        assert evaluated[i].result().info["energy"] > 0.0

    # test batched evaluation
    energies = np.array([evaluated[i].result().info["energy"] for i in range(10)])
    evaluated = hamiltonian.evaluate(dataset[:10], batch_size=3)
    for i in range(10):
        assert energies[i] == evaluated[i].result().info["energy"]

    # test equality
    hamiltonian_ = EinsteinCrystal(dataset[0], force_constant=1.1)
    assert not hamiltonian == hamiltonian_
    hamiltonian_ = EinsteinCrystal(dataset[0], force_constant=1.0)
    assert hamiltonian == hamiltonian_
    hamiltonian_ = EinsteinCrystal(dataset[1], force_constant=1.0)
    assert not hamiltonian == hamiltonian_
    assert not hamiltonian == PlumedHamiltonian(plumed_input="")

    # consider linear combination
    scaled = 0.5 * hamiltonian
    assert len(scaled) == 1
    assert scaled.get_coefficient(hamiltonian) == 0.5
    actually_scaled = EinsteinCrystal(dataset[0], force_constant=0.5)
    assert scaled.get_coefficient(actually_scaled) is None

    evaluated_scaled = scaled.evaluate(dataset[:10])
    evaluated_actually = actually_scaled.evaluate(dataset[:10])
    for i in range(1, 10):
        e = evaluated[i].result().info["energy"]
        e_scaled = evaluated_scaled[i].result().info["energy"]
        assert 0.5 * e == e_scaled
        assert 0.5 * e == evaluated_actually[i].result().info["energy"]
        assert not e == 0

    other = EinsteinCrystal(dataset[0], 4.0)
    evaluated_other = other.evaluate(dataset[:10])
    mixture = hamiltonian + other
    assert len(mixture) == 2
    assert mixture == 0.9 * other + 0.1 * other + 1.0 * hamiltonian
    _ = mixture + other
    assert mixture.get_coefficients(mixture) == (1, 1)
    assert mixture.get_coefficients(hamiltonian + actually_scaled) is None
    evaluated_mixture = mixture.evaluate(dataset[:10])
    for i in range(1, 10):
        e = evaluated[i].result().info["energy"]
        e_mixture = evaluated_mixture[i].result().info["energy"]
        e_other = evaluated_other[i].result().info["energy"]
        assert e_mixture == e + e_other

        f = evaluated[i].result().arrays["forces"]
        f_mixture = evaluated_mixture[i].result().arrays["forces"]
        f_other = evaluated_other[i].result().arrays["forces"]
        assert np.allclose(
            f_mixture,
            f + f_other,
        )

    zero = Zero()
    evaluated_zero = zero.evaluate(evaluated)
    for i in range(evaluated_zero.length().result()):
        assert "energy" not in evaluated_zero[i].result().info
    assert hamiltonian == hamiltonian + zero
    assert 2 * hamiltonian + zero == 2 * hamiltonian


def test_plumed_evaluate(context, dataset, tmp_path):
    atoms = dataset[0].result()
    center = 1
    kappa = 1
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
RESTRAINT ARG=CV AT={center} KAPPA={kappa}
""".format(
        center=center, kappa=kappa / (kJ / mol)
    )
    energy, forces, stress = evaluate_plumed(atoms, plumed_input)
    distance = atoms.get_distance(0, 1, mic=False)

    bias_energy = kappa / 2 * (distance - center) ** 2
    assert np.allclose(
        energy,
        bias_energy,
    )

    delta = atoms.positions[0, :] - atoms.positions[1, :]
    gradient = kappa * (distance - center) * delta / np.linalg.norm(delta)
    forces_ = np.zeros((len(atoms), 3))
    forces_[0, :] = -gradient
    forces_[1, :] = gradient
    assert np.allclose(
        forces,
        forces_,
    )

    # use external grid as bias, check that file is read
    hills = """#! FIELDS time CV sigma_CV height biasf
#! SET multivariate false
#! SET kerneltype gaussian
     1.00000     2.1     2.0     7  0
     2.00000     2.2     2.0     7  0
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
    for _i in range(10):
        energy, _, _ = evaluate_plumed(atoms, plumed_input)
    sigma = 2 * np.ones(2)
    height = np.array([7, 7]) * (kJ / mol)  # unit consistency
    center = np.array([2.1, 2.2])
    energy_ = np.sum(height * np.exp((distance - center) ** 2 / (-2 * sigma**2)))
    assert np.allclose(
        energy,
        energy_,
    )

    # check that hills file didn't change
    with open(path_hills, "r") as f:
        assert f.read() == hills


def test_plumed_hamiltonian(context, dataset, tmp_path):
    kappa = 1
    center = 100
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT={center} KAPPA={kappa}
""".format(
        center=center, kappa=kappa / (kJ / mol)
    )
    hamiltonian = PlumedHamiltonian(plumed_input)
    evaluated = hamiltonian.evaluate(dataset).as_list().result()
    for atoms in evaluated:
        assert np.allclose(
            atoms.info["energy"],
            kappa / 2 * (atoms.get_volume() - center) ** 2,
        )

    # use external grid as bias, check that file is read
    hills = """#! FIELDS time CV sigma_CV height biasf
#! SET multivariate false
#! SET kerneltype gaussian
     1.00000     2.1     2.0     7  0
     2.00000     2.2     2.0     7  0
"""
    path_hills = tmp_path / "hills"
    with open(path_hills, "w") as f:
        f.write(hills)
    data_future = copy_data_future(
        inputs=[File(path_hills)],
        outputs=[File(str(path_hills) + "_")],
    ).outputs[0]
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=1 SIGMA=3 HEIGHT=342 FILE={}
""".format(
        data_future.filepath
    )
    hamiltonian = PlumedHamiltonian(plumed_input, data_future)
    data = hamiltonian.evaluate(dataset)
    for i in range(data.length().result()):
        assert data[i].result().info["energy"] > 0
    hamiltonian = PlumedHamiltonian(plumed_input, File(path_hills))
    data = hamiltonian.evaluate(dataset)
    for i in range(data.length().result()):
        assert data[i].result().info["energy"] > 0


def test_json_dump(context):
    data = {
        "a": np.ones((3, 3, 3, 2)),
        "b": [1, 2, 3],
        "c": (1, 2, 4),
        "d": "asdf",
        "e": copy_app_future(False),
    }
    data_future = dump_json(
        **data,
        outputs=[psiflow.context().new_file("bla_", ".json")],
    ).outputs[0]
    psiflow.wait()
    with open(data_future.filepath, "r") as f:
        data_ = json.loads(f.read())

    new_a = np.array(data_["a"])
    assert len(new_a.shape) == 4
    assert np.allclose(
        data["a"],
        new_a,
    )
    assert data_["e"] is False
    assert type(data_["b"]) is list
    assert type(data_["c"]) is list


def test_serialization(context, dataset):
    hamiltonian = EinsteinCrystal(dataset[0], force_constant=1)
    evaluated = hamiltonian.evaluate(dataset[:3])

    # manual
    data_future = hamiltonian.serialize()
    psiflow.wait()
    calculator = deserialize(data_future.filepath)
    atoms = dataset[0].result()
    atoms.calc = calculator
    for i in range(3):
        state = dataset[i].result()
        atoms.set_positions(state.get_positions())
        atoms.set_cell(state.get_cell())
        e = atoms.get_potential_energy()
        assert e == evaluated[i].result().info["energy"]
