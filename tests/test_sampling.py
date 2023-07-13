from psiflow.data import FlowAtoms
from psiflow.walkers import RandomWalker
from psiflow.sampling import sample
from psiflow.reference import EMTReference
from psiflow.models import MACEModel


def test_sample(mace_config, dataset):
    walkers = RandomWalker.multiply(3, data_start=dataset)
    state = FlowAtoms(
            numbers=400 * np.arange(1, 4),
            positions=np.zeros((3, 3)),
            cell=np.eye(3),
            pbc=True,
            )
    walkers += RandomWalker(state, seed=10)

    reference = EMTReference()
    model = MACEModel(mace_config)
    model.initialize(dataset[:3])
    model.deploy()

    identifier = 4 
    data, identifier = sample(
            identifier,
            model,
            reference,
            walkers,
            error_thresholds_for_reset=(1e9, 1e9),
            )
