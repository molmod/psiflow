import pytest
import os
from pathlib import Path
import numpy as np

from flower.models import NequIPModel
from flower.manager import Manager, log_dataset, log_ensemble
from flower.reference import EMTReference
from flower.sampling import RandomWalker, DynamicWalker, PlumedBias
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
    manager = Manager(tmpdir, 'pytest', 'test_manager_dry_run')
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
    manager = Manager(path_output, 'pytest', 'test_manager_save_load')
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
    name = 'test'
    manager.save(name=name, model=model, ensemble=ensemble, checks=checks)
    assert (path_output / name / 'ensemble').is_dir()
    assert (path_output / name / 'ensemble' / '0').is_dir()
    assert (path_output / name / 'ensemble' / '1').is_dir()
    assert not (path_output / name / 'ensemble' / '2').is_dir()
    assert (path_output / name / 'checks').is_dir() # directory for saved checks

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
    name = 'test_'
    manager.save(name=name, model=model, ensemble=ensemble, checks=checks)
    model_, ensemble_, data_train, data_valid, checks = manager.load(
            'test_',
            context,
            )
    assert model_.config_future is not None # model was not initialized
    for check in checks: # order is arbitrary
        if type(check) == DiscrepancyCheck:
            assert check.model_old is not None
            assert check.model_new is not None


def test_log_dataset_ensemble(context, dataset, model, tmp_path):
    error_kwargs = {
            'intrinsic': False,
            'metric': 'mae',
            'properties': ['energy', 'forces', 'stress'],
            }
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
restraint: RESTRAINT ARG=CV AT=150 KAPPA=1
CV1: VOLUME
mtd: METAD ARG=CV1 PACE=1 SIGMA=10 HEIGHT=23
"""
    bias = PlumedBias(context, plumed_input)
    model.initialize(dataset[:2])
    model.deploy()
    future = log_dataset(
            'training',
            'run_name', # run name
            'test_manager_wandb', # group
            'pytest', # project
            dataset,
            visualize_structures=False,
            bias=bias,
            model=model,
            error_kwargs=error_kwargs,
            )
    future.result()

    ensemble = Ensemble.from_walker(
            RandomWalker(context, dataset[0]),
            nwalkers=10,
            dataset=dataset,
            )
    ensemble.walkers[3].tag_future = 'unsafe'
    ensemble.walkers[7].tag_future = 'unsafe'
    ensemble.biases = [None, None] + [bias.copy() for i in range(8)] # not all same bias
    future = log_ensemble(
            'run_name', # run name
            'test_manager_wandb', # group
            'pytest', # project
            ensemble,
            visualize_structures=False,
            )
    future.result()
