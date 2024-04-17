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
    reference.cell = 100 * np.eye(3)
    einstein = EinsteinCrystal(reference, force_constant=100)
    hessian = compute_harmonic(
        reference,
        einstein,
    )
    print(hessian.result())
