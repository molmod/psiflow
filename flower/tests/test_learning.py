import pytest

from flower.models import NequIPModel
from flower.learning import LearningState, Algorithm
from flower.reference import EMTReference
from flower.sampling import RandomWalker, Ensemble

from common import context, nequip_config
from test_dataset import dataset


@pytest.fixture
def learning_state(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config)
    reference = EMTReference(context)
    walker = RandomWalker(context, dataset[0])
    ensemble = Ensemble.from_walker(walker, nwalkers=2)
    learning_state = LearningState(model=model, reference=reference, ensemble=ensemble)
    return learning_state


def test_learning_state(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config)
    with pytest.raises(TypeError):
        state = LearningState(model=model)
    reference = EMTReference(context)
    walker = RandomWalker(context, dataset[0])
    ensemble = Ensemble.from_walker(walker, nwalkers=2)
    learning_state = LearningState(
            model=model,
            reference=reference,
            ensemble=ensemble,
            )


def test_algorithm_write(learning_state, tmpdir):
    algorithm = Algorithm(tmpdir)
    algorithm.write(learning_state, iteration=3)
