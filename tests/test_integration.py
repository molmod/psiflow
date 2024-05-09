import numpy as np
from ase.units import kB

from psiflow.data import check_equality
from psiflow.free_energy import Integration
from psiflow.hamiltonians import EinsteinCrystal, Harmonic
from psiflow.tools import optimize, compute_harmonic
from psiflow.utils import multiply


def test_integration_simple(dataset):
    dataset = dataset[:10]
    einstein = EinsteinCrystal(dataset[1], force_constant=2)
    geometry = optimize(
        dataset[3],
        einstein,
        steps=20000,
        ftol=1e-4,
    )
    hessian = compute_harmonic(
        geometry,
        einstein,
        pos_shift=5e-4,
    )
    harmonic = Harmonic(geometry, hessian)

    integration = Integration(
        harmonic,
        temperatures=[300, 400],
        delta_hamiltonian=(-0.1) * harmonic,
        delta_coefficients=np.array([0.0, 0.5, 1.0]),
    )
    walkers = integration.create_walkers(
        dataset,
        initialize_by="quench",
    )
    for walker in walkers:
        assert check_equality(walker.start, dataset[1]).result()

    assert len(integration.states) == 6

    integration.sample(steps=100, step=6)
    integration.compute_gradients()
    for i, state in enumerate(integration.states):
        assert state.gradients["delta"] is not None
        assert state.gradients["temperature"] is not None

        # manual computation of delta gradient
        delta = -0.1 * harmonic
        energies = delta.evaluate(integration.outputs[i].trajectory).get("energy")
        assert np.allclose(
            state.gradients["delta"].result(),
            np.mean(energies.result()) / (kB * state.temperature),
        )

    #frequencies = compute_frequencies(hessian, geometry)
    F0 = harmonic.compute_free_energy(temperature=300).result()
    scaled = Harmonic(multiply(hessian, 0.9), geometry)
    F1 = scaled.compute_free_energy(temperature=300).result()

    integrated = integration.along_delta(temperature=300).result()
    assert len(integrated) == 3
    print("\nalong delta")
    print("   computed delta_F: {}".format(integrated[-1]))
    print("theoretical delta_F: {}".format(F1 - F0))
    print("")

    # integrated = integration.along_temperature(delta_coefficient=1.0).result()
    # assert len(integrated) == 2
    # assert np.allclose(integrated[0], 0.0)
    # F2 = np.sum(compute_free_energy(frequencies, 400).result())
    # print('\nalong temperature')
    # print('   computed delta_F: {}'.format(integrated[-1] / (kB * 400)))
    # print('theoretical delta_F: {}'.format(F2 / (kB * 400) - F1 / (kB * 300)))
