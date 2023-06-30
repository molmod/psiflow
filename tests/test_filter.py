import numpy as np

from psiflow.filter import Committee
from psiflow.models import MACEModel


def test_committee_mace(context, mace_config, dataset):
    mace_config['max_num_epochs'] = 1
    models = [MACEModel(mace_config) for i in range(3)]
    committee = Committee(models)
    for i, model in enumerate(committee.models):
        assert model.seed == i
    committee.train(dataset[:5], dataset[5:10])
    extracted, disagreements = committee.apply(dataset, 5)
    assert extracted.length().result() == 5
    assert np.all(disagreements.result() > 0)
    index_max = int(np.argmax(disagreements.result()))
    assert np.allclose(
            extracted[-1].result().positions,
            dataset[index_max].result().positions,
            )
