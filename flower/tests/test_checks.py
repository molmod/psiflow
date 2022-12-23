
from flower.checks import InteratomicDistanceCheck, DiscrepancyCheck, \
        SafetyCheck
from flower.models import NequIPModel
from flower.sampling import RandomWalker

from common import context, nequip_config
from test_dataset import dataset


def test_distance_check(context, dataset):
    check = InteratomicDistanceCheck(threshold=0.1)
    assert check(dataset[0]).result() is not None
    check = InteratomicDistanceCheck(threshold=10)
    assert check(dataset[0]).result() is None


def test_discrepancy_check(context, dataset, nequip_config):
    model_old = NequIPModel(context, nequip_config, dataset[:5])
    model_old.deploy()
    model_new = NequIPModel(context, nequip_config, dataset[-5:])
    model_new.deploy()
    check = DiscrepancyCheck(
            model_old,
            model_new,
            metric='mae',
            properties=['energy', 'forces'],
            thresholds=[0.01, 0.001], # will be exceeded
            )
    assert check(dataset[6]).result() is not None
    check.thresholds = [100, 10] # mae's should be lower than this
    assert check(dataset[6]).result() is None


def test_safety(context, dataset):
    walker = RandomWalker(context, dataset[0])
    state = walker.propagate()
    check = SafetyCheck()
    assert check(state, walker.tag_future).result() is not None
    walker.tag_future = 'unsafe'
    assert check(state, walker.tag_future).result() is None

