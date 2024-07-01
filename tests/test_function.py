import numpy as np
from ase.units import kJ, mol

from psiflow.functions import EinsteinCrystalFunction, PlumedFunction, HarmonicFunction
from psiflow.hamiltonians import EinsteinCrystal, PlumedHamiltonian, Harmonic, Zero


def test_einstein_crystal(dataset):
    function = EinsteinCrystalFunction(
        force_constant=1,
        centers=dataset[0].result().per_atom.positions,
        volume=0.0,
    )

    nstates = 4
    geometries = dataset[:nstates].reset().geometries().result()
    energy, forces, stress = function(geometries).values()
    assert np.all(energy >= 0)
    assert energy[0] == 0
    assert np.allclose(  # forces point to centers
        forces,
        function.centers.reshape(1, -1, 3) - dataset[:4].get('positions').result(),
    )
    assert geometries[0].energy is None
    hamiltonian = EinsteinCrystal(dataset[0], force_constant=1)

    forces_, stress_, energy_ = hamiltonian.compute(dataset[:4], outputs=['forces', 'stress', 'energy'])
    assert np.allclose(
        energy_.result(),
        energy,
    )
    assert np.allclose(
        forces_.result(),
        forces,
    )

    forces = hamiltonian.compute(dataset[:4], outputs=['forces'], batch_size=3)
    assert np.allclose(
        forces.result(),
        forces_.result(),
    )

    # hamiltonian = EinsteinCrystal(
    #     force_constant=function.force_constant,
    #     centers=function.centers,
    # )
    # data = dataset[:10].reset()
    # evaluated = data.evaluate(hamiltonian, outputs=['energy'], batch_size=None)
    # evaluated_ = data.evaluate(hamiltonian, outputs=['energy', 'forces'], batch_size=3)
    # for i, geometry in enumerate(evaluated.geometries().result()):
    #     assert np.all(np.isnan(geometry.per_atom.forces))
    #     assert np.allclose(geometry.energy, energy.result()[i])
    #     assert np.allclose(
    #         evaluated_[i].result().energy,
    #         geometry.energy,
    #     )


def test_plumed_function(tmp_path, dataset, dataset_h2):
    data = dataset + dataset_h2
    plumed_str = """
D1: DISTANCE ATOMS=1,2 NOPBC
CV: BIASVALUE arg=D1
"""
    function = PlumedFunction(plumed_str)
    outputs = function(data.geometries().result())

    f = 1 / (kJ / mol) * 10  # eV --> kJ/mol and nm --> A
    positions = data.get('positions').result()
    manual = np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=1)
    assert np.allclose(
        outputs['energy'] * f,
        manual,
    )
    gradient = (positions[:, 0, :] - positions[:, 1, :]) / manual.reshape(-1, 1)
    assert np.allclose(
        outputs['forces'][:, 0, :] * f,
        gradient * (-1.0),
    )

    # use external grid as bias, check that file is read
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=50 KAPPA=1
"""
    function = PlumedFunction(plumed_input)
    energy, forces, stress = function(dataset.geometries().result()).values()

    volumes = np.linalg.det(dataset.get('cell').result())
    energy_ = (volumes - 50) ** 2 * (kJ / mol) / 2
    assert np.allclose(
        energy,
        energy_,
    )

    hamiltonian = PlumedHamiltonian(plumed_input)
    energy_, forces_, stress_ = hamiltonian.compute(dataset)

    assert np.allclose(energy, energy_.result())
    assert np.allclose(stress, stress_.result())


def test_harmonic_function(dataset):
    reference = dataset[0].result()
    function = HarmonicFunction(
        reference.per_atom.positions,
        np.eye(3 * len(reference)),
        reference.energy,
    )
    einstein = EinsteinCrystalFunction(1.0, reference.per_atom.positions)

    energy, forces, _ = function(dataset[:10].geometries().result()).values()
    energy_, forces_, _ = einstein(dataset[:10].geometries().result()).values()

    assert np.allclose(energy - reference.energy, energy_)
    assert np.allclose(forces_, forces)

    harmonic = Harmonic(dataset[0], np.eye(3 * len(reference)))

    energy, forces, _ = harmonic.compute(dataset[:10])
    assert np.allclose(energy.result() - reference.energy, energy_)
    assert np.allclose(forces.result(), forces_)


def test_hamiltonian_arithmetic(dataset):
    hamiltonian = EinsteinCrystal(dataset[0], force_constant=1)
    hamiltonian_ = EinsteinCrystal(dataset[0].result(), force_constant=1.1)
    assert not hamiltonian == hamiltonian_
    hamiltonian_ = EinsteinCrystal(dataset[0], force_constant=1)
    assert hamiltonian != hamiltonian_  # app future copied
    hamiltonian_.reference_geometry = hamiltonian.reference_geometry
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

    energy_scaled = scaled.compute(dataset[:10], ['energy'])
    energy_actually = actually_scaled.compute(dataset[:10], ['energy'])
    assert np.allclose(energy_scaled.result(), energy_actually.result())

    energy, forces, _ = hamiltonian.compute(dataset[:10])
    other = EinsteinCrystal(dataset[0], 4.0)
    mixture = hamiltonian + other
    assert len(mixture) == 2
    assert mixture == 0.9 * other + 0.1 * other + 1.0 * hamiltonian
    _ = mixture + other
    assert mixture.get_coefficients(mixture) == (1, 1)
    assert mixture.get_coefficients(hamiltonian + actually_scaled) is None
    energy_, forces_, _ = mixture.compute(dataset[:10])
    assert np.allclose(energy_.result(), 5 * energy.result())
    assert np.allclose(forces_.result(), 5 * forces.result())

    zero = Zero()
    energy, forces, stress = zero.compute(dataset[:10])
    assert np.allclose(energy.result(), 0.0)
    assert hamiltonian == hamiltonian + zero
    assert 2 * hamiltonian + zero == 2 * hamiltonian
