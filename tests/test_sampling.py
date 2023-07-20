import numpy as np

from psiflow.data import FlowAtoms, Dataset
from psiflow.walkers import RandomWalker, DynamicWalker
from psiflow.reference import EMTReference
from psiflow.models import MACEModel
from psiflow.sampling import sample_with_model, sample_with_committee
from psiflow.committee import Committee


def test_sample_model(mace_config, dataset):
    walkers = RandomWalker.multiply(3, data_start=dataset)
    state = FlowAtoms(
            numbers=100 + np.arange(1, 4),
            positions=np.zeros((3, 3)),
            cell=np.eye(3),
            pbc=True,
            )
    walkers.append(RandomWalker(state, seed=10))
    walkers.append(DynamicWalker(dataset[0], steps=100, step=1, start=0))
    walkers.append(DynamicWalker(dataset[0], steps=10, step=1, start=0, force_threshold=1e-7))

    reference = EMTReference()
    model = MACEModel(mace_config)
    model.initialize(dataset[:3] + Dataset([state]))
    model.deploy()

    assert not np.allclose(dataset[0].result().arrays['forces'], 0.0)

    identifier = 4 
    data, identifier = sample_with_model(
            model,
            reference,
            walkers,
            identifier,
            error_thresholds_for_reset=(1e9, 1e9),
            )
    assert data.length().result() == 4 # one state should have failed
    for i in range(4):
        assert data[i].result().reference_status # should be successful
        assert 'identifier' in data[i].result().info.keys()
        assert data[i].result().info['identifier'] >= 4
        assert data[i].result().info['identifier'] <= 7
        

def test_sample_committee(mace_config, dataset):
    walkers = RandomWalker.multiply(3, data_start=dataset)
    walkers.append(DynamicWalker(dataset[0], steps=100, step=1, start=0))
    walkers.append(DynamicWalker(dataset[0], steps=10, step=1, start=0, force_threshold=1e-7))

    reference = EMTReference()
    mace_config['max_num_epochs'] = 1
    models = [MACEModel(mace_config) for i in range(4)]
    committee = Committee(models)
    committee.train(dataset[:5], dataset[5:10])

    identifier = 0 
    data, identifier = sample_with_committee(
            committee,
            reference,
            walkers,
            identifier,
            nstates=3
            )
    for i in range(3):
        assert data[i].result().reference_status # should be successful
        assert 'identifier' in data[i].result().info.keys()
        assert data[i].result().info['identifier'] <= 2
