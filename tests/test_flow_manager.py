import pytest
import logging
import os
from pathlib import Path
import wandb
import numpy as np

from psiflow.data import Dataset
from psiflow.models import NequIPModel
from psiflow.experiment import FlowManager, log_data, log_generators
from psiflow.reference import EMTReference
from psiflow.sampling import RandomWalker, DynamicWalker, PlumedBias
from psiflow.generator import Generator
from psiflow.checks import SafetyCheck, DiscrepancyCheck, \
        InteratomicDistanceCheck
from psiflow.utils import log_data_to_wandb


@pytest.fixture
def generators(context, dataset):
    walker = RandomWalker(context, dataset[0])
    generators = Generator('random', walker).multiply(2)
    return generators


@pytest.fixture
def reference(context):
    return EMTReference(context)


def test_log_dataset_generators(context, dataset, nequip_config, tmp_path, reference):
    error_kwargs = {
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
    model = NequIPModel(context, nequip_config)
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
            'test_log_dataset_generators',
            'pytest',
            'CV',
            ['training'],
            inputs=[future],
            )
    log0.result()

    generators = Generator(
            'random',
            RandomWalker(context, dataset[0]),
            bias,
            ).multiply(10)
    generators[3].walker.tag_unsafe()
    generators[7].walker.tag_unsafe()
    generators[0].bias = None
    generators[1].bias = None
    states = [g(model, reference, None) for g in generators]
    for state in states:
        state.result()
    future = log_generators(generators)
    log1 = log_data_to_wandb(
            'run_name',
            'test_log_dataset_generators',
            'pytest',
            'dummy',
            ['generators'],
            inputs=[future],
            )
    log1.result()


def test_flow_manager_save_load(context, dataset, nequip_config, generators, tmp_path):
    model = NequIPModel(context, nequip_config)
    path_output = Path(tmp_path) / 'parsl_internal'
    path_output.mkdir()
    generators = (
            Generator('random', RandomWalker(context, dataset[0])).multiply(2) +
            Generator('dynamic', DynamicWalker(context, dataset[1])).multiply(2)
            )
    flow_manager = FlowManager(path_output, 'pytest', 'test_flow_manager_save_load')
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
    flow_manager.save(
            name=name,
            model=model,
            generators=generators,
            checks=checks,
            data_failed=dataset,
            )
    assert (path_output / name / 'generators').is_dir()
    assert (path_output / name / 'generators' / 'random0').is_dir()
    assert (path_output / name / 'generators' / 'random1').is_dir()
    assert (path_output / name / 'generators' / 'dynamic0').is_dir()
    assert (path_output / name / 'generators' / 'dynamic1').is_dir()
    assert not (path_output / name / 'generators' / '0').is_dir()
    assert (path_output / name / 'checks').is_dir() # directory for saved checks
    assert (path_output / name / 'failed.xyz').is_file()

    model_, generators_, data_train, data_valid, checks = flow_manager.load(
            'test',
            context,
            )
    assert model_.config_future is None # model was not initialized
    assert len(generators_) == 4
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
    flow_manager.save(name=name, model=model, generators=generators, checks=checks)
    model_, generators_, data_train, data_valid, checks = flow_manager.load(
            'test_',
            context,
            )
    assert model_.config_future is not None # model was not initialized
    for check in checks: # order is arbitrary
        if type(check) == DiscrepancyCheck:
            assert check.model_old is not None
            assert check.model_new is not None


def test_flow_manager_dry_run(
        context,
        dataset,
        nequip_config,
        generators,
        reference,
        tmp_path,
        #caplog,
        ):
    #caplog.set_level(logging.INFO)
    model = NequIPModel(context, nequip_config)
    path_output = Path(tmp_path) / 'parsl_internal'
    path_output.mkdir()
    flow_manager = FlowManager(path_output, 'pytest', 'test_flow_manager_dry_run')
    with pytest.raises(AssertionError):
        flow_manager.dry_run(model, reference) # specify either walker or ensemble
    random_walker = RandomWalker(context, dataset[0])
    flow_manager.dry_run(model, reference, random_walker=random_walker)
