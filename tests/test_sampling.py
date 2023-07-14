import numpy as np

from psiflow.data import FlowAtoms
from psiflow.walkers import RandomWalker
from psiflow.reference import EMTReference
from psiflow.models import MACEModel
from psiflow.sampling import sample


def test_sample(mace_config, dataset):
    walkers = RandomWalker.multiply(3, data_start=dataset)
    state = FlowAtoms(
            numbers=100 + np.arange(1, 4),
            positions=np.zeros((3, 3)),
            cell=np.eye(3),
            pbc=True,
            )
    walkers.append(RandomWalker(state, seed=10))

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
    assert data.length().result() == 3 # one state should have failed
    for i in range(3):
        assert data[i].result().reference_status # should be successful
        assert 'identifier' in data[i].result().info.keys()
        assert data[i].result().info['identifier'] >= 4
        assert data[i].result().info['identifier'] <= 6
