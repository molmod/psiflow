import numpy as np
from ase.units import _c, second

from psiflow.hamiltonians import EinsteinCrystal, get_mace_mp0
from psiflow.tools import compute_harmonic, optimize
from psiflow.tools.utils import compute_frequencies


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
        asr="poly",
    )
    frequencies = compute_frequencies(hessian, geometry).result()
    # check that highest frequency in inv cm corresponds to 3500 - 4000
    frequency_invcm = (frequencies[-1] * second) / (_c * 1e2)  # in invcm
    assert np.abs(frequency_invcm - 4000) < 1000
