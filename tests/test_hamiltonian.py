import numpy as np

from psiflow.hamiltonians import EinsteinCrystal


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
