import pytest
import os
from pathlib import Path
import numpy as np

from flower.models import NequIPModel
from flower.learning import Manager
from flower.reference import EMTReference
from flower.sampling import RandomWalker, DynamicWalker
from flower.ensemble import Ensemble
from flower.checks import SafetyCheck, DiscrepancyCheck, \
        InteratomicDistanceCheck


@pytest.fixture
def model(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config)
    return model


@pytest.fixture
def ensemble(context, dataset):
    walker = RandomWalker(context, dataset[0])
    ensemble = Ensemble.from_walker(walker, nwalkers=2)
    return ensemble


@pytest.fixture
def reference(context):
    return EMTReference(context)


def test_manager_dry_run(context, dataset, model, ensemble, reference, tmpdir):
    manager = Manager(tmpdir)
    with pytest.raises(AssertionError):
        manager.dry_run(model, reference) # specify either walker or ensemble
    random_walker = RandomWalker(context, dataset[0])
    manager.dry_run(model, reference, random_walker=random_walker)


def test_manager_save_load(context, dataset, model, ensemble, tmpdir):
    path_output = Path(tmpdir)
    walkers = []
    walkers.append(RandomWalker(context, dataset[0]))
    walkers.append(DynamicWalker(context, dataset[1]))
    ensemble = Ensemble(context, walkers=walkers, biases=[None, None])
    manager = Manager(path_output)
    checks = [
            SafetyCheck(),
            DiscrepancyCheck(
                metric='mae',
                properties=['energy'],
                thresholds=[1],
                model_old=None,
                model_new=None,
                ),
            ]
    prefix = 'test'
    manager.save(prefix=prefix, model=model, ensemble=ensemble, checks=checks)
    assert (path_output / prefix / 'ensemble').is_dir()
    assert (path_output / prefix / 'ensemble' / '0').is_dir()
    assert (path_output / prefix / 'ensemble' / '1').is_dir()
    assert not (path_output / prefix / 'ensemble' / '2').is_dir()
    assert (path_output / prefix / 'checks').is_dir() # directory for saved checks
    print(os.listdir(path_output / prefix))

    model_, ensemble_, data_train, data_valid, checks = manager.load(
            'test',
            context,
            )
    assert model_.config_future is None # model was not initialized
    assert np.allclose(
            ensemble.walkers[0].state_future.result().positions,
            ensemble_.walkers[0].state_future.result().positions,
            )
    assert np.allclose(
            ensemble.walkers[1].state_future.result().positions,
            ensemble_.walkers[1].state_future.result().positions,
            )
    assert data_train.length().result() == 0
    assert data_valid.length().result() == 0
    assert len(checks) == 2
    model.initialize(dataset[:2])
    model.deploy()
    checks = [
            SafetyCheck(),
            DiscrepancyCheck(
                metric='mae',
                properties=['energy'],
                thresholds=[1],
                model_old=model,
                model_new=model.copy(),
                ),
            ]
    prefix = 'test_'
    manager.save(prefix=prefix, model=model, ensemble=ensemble, checks=checks)
    model_, ensemble_, data_train, data_valid, checks = manager.load(
            'test_',
            context,
            )
    assert model_.config_future is not None # model was not initialized
    assert checks[1].model_old is not None
    assert checks[1].model_new is not None
