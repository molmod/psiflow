import numpy as np
from ase.units import _c, second

from psiflow.hamiltonians import EinsteinCrystal, get_mace_mp0
from psiflow.hamiltonians._harmonic import compute_frequencies, harmonic_free_energy
from psiflow.tools import compute_harmonic, optimize, optimize_dataset


def test_optimize(dataset):
    einstein = EinsteinCrystal(dataset[2], force_constant=100)
    final = optimize(dataset[0], einstein, steps=1000000).result()

    assert np.allclose(
        final.per_atom.positions,
        dataset[2].result().per_atom.positions,
        atol=1e-4,
    )
    # assert np.allclose(
    #        final.cell,
    #        dataset[2].result().cell,
    #        atol=1e-4,
    #        )
    assert np.allclose(final.energy, 0.0)  # einstein energy >= 0

    optimized = optimize_dataset(dataset[:3], einstein, steps=1000000)
    for g in optimized.geometries().result():
        assert np.allclose(g.energy, 0.0)


def test_phonons(dataset):
    reference = dataset[2].result()
    constant = 10
    einstein = EinsteinCrystal(reference, force_constant=constant)

    hessian = compute_harmonic(
        reference,
        einstein,
        asr="none",  # einstein == translationally VARIANT
    )
    assert np.allclose(
        hessian.result(),
        constant * np.eye(3 * len(reference)),
    )


def test_dihydrogen(dataset_h2):
    geometry = dataset_h2[0].result()
    geometry.cell = 20 * np.eye(3)
    hamiltonian = get_mace_mp0("small")
    optimized = optimize(
        geometry,
        hamiltonian,
        steps=2000,
        ftol=1e-4,
    ).result()
    assert optimized.energy is not None
    assert np.linalg.norm(optimized.per_atom.forces) < 1e-4
    hessian = compute_harmonic(
        optimized,
        hamiltonian,
        asr="crystal",
        pos_shift=0.001,
    )
    frequencies = compute_frequencies(hessian, geometry).result()
    # check that highest frequency in inv cm corresponds to 3500 - 4000
    frequencies_invcm = (frequencies * second) / (_c * 1e2)  # in invcm
    assert np.abs(frequencies_invcm[-1] - 4000) < 1000


def test_frequency_oscillator():
    for quantum in [True, False]:
        f0 = harmonic_free_energy(1.0, 300, quantum=quantum).result()
        f1 = harmonic_free_energy(1.1, 300, quantum=quantum).result()
        assert f1 > f0

        f2 = harmonic_free_energy(1.0, 400, quantum=quantum).result()
        assert f0 > f2
