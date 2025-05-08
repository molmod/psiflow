import numpy as np
from ase.units import _c, kB, second

from psiflow.free_energy import (
    Integration,
    compute_frequencies,
    compute_harmonic,
    harmonic_free_energy,
)
from psiflow.geometry import check_equality
from psiflow.hamiltonians import EinsteinCrystal, Harmonic, MACEHamiltonian
from psiflow.sampling.ase import optimize


def test_integration_simple(dataset):
    dataset = dataset[:10]
    einstein = EinsteinCrystal(dataset[1], force_constant=2)
    geometry = optimize(
        dataset[3],
        einstein,
        mode='fix_cell',
        f_max=1e-4,
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
        energies = delta.compute(integration.outputs[i].trajectory, "energy")
        assert np.allclose(
            state.gradients["delta"].result(),
            np.mean(energies.result()) / (kB * state.temperature),
        )

    hessian = hessian.result()
    frequencies0 = compute_frequencies(hessian, geometry)
    frequencies1 = compute_frequencies(hessian * 0.9, geometry)
    F0 = harmonic_free_energy(frequencies0, 300).result()
    F1 = harmonic_free_energy(frequencies1, 300).result()

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


def test_integration_temperature(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=1)
    integration = Integration(
        hamiltonian=einstein,
        temperatures=[300, 400],
        pressure=0.0,
    )
    integration.create_walkers(dataset[:3])
    integration.sample(steps=10, step=1)
    integration.compute_gradients()
    gradient0 = integration.states[0].gradients["temperature"]

    integration = Integration(
        hamiltonian=einstein,
        temperatures=[300, 400],
    )
    integration.create_walkers(dataset[:3])
    integration.sample(steps=10, step=1)
    integration.compute_gradients()
    gradient1 = integration.states[0].gradients["temperature"]
    assert np.allclose(gradient0.result(), gradient1.result())


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
        hessian.result(), constant * np.eye(3 * len(reference)), rtol=1e-4
    )


def test_dihydrogen(dataset_h2):
    geometry = dataset_h2[0].result()
    geometry.cell = 20 * np.eye(3)
    hamiltonian = MACEHamiltonian.mace_mp0("small")
    optimized = optimize(
        geometry,
        hamiltonian,
        mode='fix_cell',
        f_max=1e-4,
    ).result()
    assert optimized.energy is not None
    assert np.linalg.norm(optimized.per_atom.forces) < 1e-2
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
