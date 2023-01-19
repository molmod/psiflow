from pathlib import Path

from psiflow.checks import InteratomicDistanceCheck, DiscrepancyCheck, \
        SafetyCheck, load_checks
from psiflow.models import NequIPModel
from psiflow.sampling import RandomWalker


def test_distance_check(context, dataset, tmp_path):
    check = InteratomicDistanceCheck(threshold=0.1)
    assert check(dataset[0]).result() is not None
    assert check.npasses.result() == 1
    assert len(check.states.result()) == 0
    check = InteratomicDistanceCheck(threshold=10)
    assert check(dataset[0]).result() is None
    assert len(check.states.result()) == 1 # contains one AppFuture
    path = Path(tmp_path)
    check.save(path)
    check_ = load_checks(path, context)[0]
    assert type(check_) == InteratomicDistanceCheck
    assert check_.threshold == 10


# test occasionally fails due to torch.jit compilation errors
def test_discrepancy_check(context, dataset, nequip_config, tmp_path):
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
            thresholds=[1, 10], # will be exceeded (meV/atom, meV/angstrom)
            model_old=model_old,
            model_new=model_new,
            )
    assert check(dataset[6]).result() is not None
    assert check.nchecks == 1
    assert check.npasses.result() == 1
    assert len(check.states.result()) == 0
    check.thresholds = [1e4, 1e4] # mae's should be lower than this
    assert check(dataset[6]).result() is None
    assert check.nchecks == 2
    assert check.npasses.result() == 1
    assert len(check.states.result()) == 1
    path = Path(tmp_path)
    check.save(path)
    check_ = load_checks(path, context)[0]
    assert type(check_) == DiscrepancyCheck
    assert check_.model_old.config_raw['seed'] == 123
    assert check_.model_new.config_raw['seed'] == 111


def test_safety_check(context, dataset, tmp_path):
    walker = RandomWalker(context, dataset[0])
    state = walker.propagate()
    check = SafetyCheck()
    assert check(state, walker.tag_future).result() is not None
    assert check.nchecks == 1
    assert check.npasses.result() == 1
    walker.tag_unsafe()
    assert check(state, walker.tag_future).result() is None
    assert check.nchecks == 2
    assert check.npasses.result() == 1
    path = Path(tmp_path)
    check.save(path)
    check_ = load_checks(path, context)[0]
    assert type(check_) == SafetyCheck
