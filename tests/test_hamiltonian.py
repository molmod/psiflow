import numpy as np
from ase.units import kJ, mol

from psiflow.hamiltonians import EinsteinCrystal, PlumedHamiltonian
from psiflow.hamiltonians._plumed import evaluate_plumed


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
    actually_scaled = EinsteinCrystal(dataset[0], force_constant=0.5)
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


def test_plumed_hamiltonian(context, dataset):
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
    for i, atoms in enumerate(evaluated):
        assert np.allclose(
            atoms.info["energy"],
            kappa / 2 * (atoms.get_volume() - center) ** 2,
        )
