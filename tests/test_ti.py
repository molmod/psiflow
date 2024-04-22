from psiflow.data import check_equality
from psiflow.free_energy import Integration
from psiflow.hamiltonians import EinsteinCrystal, Harmonic
from psiflow.tools import compute_harmonic, optimize


def test_integration_simple(dataset_h2):
    einstein = EinsteinCrystal(dataset_h2[1], force_constant=1)
    geometry = optimize(
        dataset_h2[3],
        einstein,
        steps=20000,
        ftol=1e-4,
    )
    hessian = compute_harmonic(
        geometry,
        einstein,
        pos_shift=1e-3,
    )
    harmonic = Harmonic(geometry, hessian)

    integration = Integration(
        harmonic,
        temperatures=[300],
        delta=(2 * harmonic - harmonic),
        npoints=3,
    )
    walkers = integration.create_walkers(
        dataset_h2,
        initialize_by="quench",
    )
    for walker in walkers:
        assert check_equality(walker.start, dataset_h2[1]).result()

    assert len(integration.states) == 3

    integration.sample(steps=50, step=4)
    integration.compute_gradients()

    for state in integration.states:
        assert state.gradients["lambda"] is not None
        # assert np.allclose(
        #        state.gradients['lambda'].result(),
        #        #harmonic.evaluate(
