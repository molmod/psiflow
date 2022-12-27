from pathlib import Path

from flower.checks import InteratomicDistanceCheck, DiscrepancyCheck, \
        SafetyCheck, load_checks
from flower.models import NequIPModel
from flower.sampling import RandomWalker

from common import context, nequip_config
from test_dataset import dataset


def test_distance_check(context, dataset, tmpdir):
    check = InteratomicDistanceCheck(threshold=0.1)
    assert check(dataset[0]).result() is not None
    check = InteratomicDistanceCheck(threshold=10)
    assert check(dataset[0]).result() is None
    path = Path(tmpdir)
    check.save(path)
    check_ = load_checks(path, context)[0]
    assert type(check_) == InteratomicDistanceCheck
    assert check_.threshold == 10


def test_discrepancy_check(context, dataset, nequip_config, tmpdir):
    model_old = NequIPModel(context, nequip_config)
    model_old.set_seed(123)
    model_old.initialize(dataset[5:])
    model_old.deploy()
    model_new = NequIPModel(context, nequip_config)
    model_new.set_seed(111) # networks only substantially different with different seeds
    model_new.initialize(dataset[-5:])
    model_new.deploy()
    check = DiscrepancyCheck(
            metric='mae',
            properties=['energy', 'forces'],
            thresholds=[0.01, 0.001], # will be exceeded
            model_old=model_old,
            model_new=model_new,
            )
    assert check(dataset[6]).result() is not None
    check.thresholds = [100, 10] # mae's should be lower than this
    assert check(dataset[6]).result() is None
    path = Path(tmpdir)
    check.save(path, context)
    check_ = load_checks(path, context)[0]
    assert type(check_) == DiscrepancyCheck
    assert check_.model_old.config_raw['seed'] == 123
    assert check_.model_new.config_raw['seed'] == 111


def test_safety(context, dataset, tmpdir):
    walker = RandomWalker(context, dataset[0])
    state = walker.propagate()
    check = SafetyCheck()
    assert check(state, walker.tag_future).result() is not None
    walker.tag_future = 'unsafe'
    assert check(state, walker.tag_future).result() is None
    path = Path(tmpdir)
    check.save(path, context)
    check_ = load_checks(path, context)[0]
    assert type(check_) == SafetyCheck
