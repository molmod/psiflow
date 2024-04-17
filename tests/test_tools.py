import numpy as np

from psiflow.hamiltonians import EinsteinCrystal
from psiflow.tools import compute_harmonic, optimize


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
