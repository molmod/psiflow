import numpy as np

from psiflow.committee import Committee
from psiflow.models import MACEModel


def test_committee_mace(gpu, mace_config, dataset):
    mace_config["max_num_epochs"] = 1
    models = [MACEModel(mace_config) for i in range(4)]
    committee = Committee(models)
    for i, model in enumerate(committee.models):
        assert model.seed == i
    committee.train(dataset[:5], dataset[5:10])
    disagreements = committee.compute_disagreements(dataset)
    assert np.all(disagreements.result() > 0)
    index_max = int(np.argmax(disagreements.result()))

    extracted, disagreements_ = committee.apply(dataset, 4)
    assert np.allclose(
        disagreements_.result(),
        disagreements.result(),
    )
    assert extracted.length().result() == 4
    assert np.allclose(
        extracted[0].result().positions,
        dataset[index_max].result().positions,
    )
