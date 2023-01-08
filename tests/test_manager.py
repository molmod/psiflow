import pytest
import os
from pathlib import Path
import wandb
import numpy as np

from flower.models import NequIPModel
from flower.manager import Manager, log_data, log_ensemble
from flower.reference import EMTReference
from flower.sampling import RandomWalker, DynamicWalker, PlumedBias
from flower.ensemble import Ensemble
from flower.checks import SafetyCheck, DiscrepancyCheck, \
        InteratomicDistanceCheck
from flower.utils import log_data_to_wandb


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
    wandb_id = wandb.util.generate_id()
    future = log_data(
            dataset,
            bias=bias,
            model=model,
            error_kwargs=error_kwargs,
            )
    log0 = log_data_to_wandb(
            'run_name',
            'test_log_dataset_ensemble',
            'pytest',
            'CV',
            ['training'],
            inputs=[future],
            )

    ensemble = Ensemble.from_walker(
            RandomWalker(context, dataset[0]),
            nwalkers=10,
            dataset=dataset,
            )
    ensemble.walkers[3].tag_future = 'unsafe'
    ensemble.walkers[7].tag_future = 'unsafe'
    ensemble.biases = [None, None] + [bias.copy() for i in range(8)] # not all same bias
    checks = [SafetyCheck(), InteratomicDistanceCheck(threshold=0.6)]
    dataset = ensemble.sample(10, checks=checks)
    dataset.data_future.result() # force execution of join_app
    assert len(checks[0].states.result()) == 2
    future = log_ensemble(ensemble)
    log1 = log_data_to_wandb(
            'run_name',
            'test_log_dataset_ensemble',
            'pytest',
            'dummy',
            ['ensemble'],
            inputs=[future],
            )
    log1.result()
    log0.result()


def test_manager_save_load(context, dataset, model, ensemble, tmp_path):
    path_output = Path(tmp_path)
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


def test_manager_dry_run(context, dataset, model, ensemble, reference, tmp_path):
    manager = Manager(tmp_path, 'pytest', 'test_manager_dry_run')
    with pytest.raises(AssertionError):
        manager.dry_run(model, reference) # specify either walker or ensemble
    random_walker = RandomWalker(context, dataset[0])
    manager.dry_run(model, reference, random_walker=random_walker)
