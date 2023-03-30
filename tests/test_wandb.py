import pytest
import logging
import os
import wandb
from pathlib import Path
import numpy as np

from psiflow.data import Dataset
from psiflow.generate import generate_all
from psiflow.models import NequIPModel, MACEModel
from psiflow.wandb_utils import WandBLogger, log_data, to_wandb
from psiflow.reference import EMTReference
from psiflow.sampling import RandomWalker, DynamicWalker, PlumedBias, \
        BiasedDynamicWalker
from psiflow.checks import SafetyCheck, DiscrepancyCheck, \
        InteratomicDistanceCheck


@pytest.fixture
def reference(context):
    return EMTReference()


def test_log_dataset_walkers(context, dataset, nequip_config, tmp_path, reference):
    error_kwargs = {
            'metric': 'mae',
            'properties': ['energy', 'forces', 'stress'],
            }
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
restraint: RESTRAINT ARG=CV AT=150 KAPPA=1
CV1: MATHEVAL ARG=CV VAR=a FUNC=2*a PERIODIC=NO
mtd: METAD ARG=CV1 PACE=1 SIGMA=10 HEIGHT=23
"""
    bias = PlumedBias(plumed_input)
    model = NequIPModel(nequip_config)
    model.initialize(dataset[:2])
    model.deploy()
    future = log_data(
            bias.evaluate(dataset, as_dataset=True),
            model=model,
            error_x_axis='CV1',
            error_kwargs=error_kwargs,
            )
    log0 = to_wandb(
            'run_name',
            'test_log_dataset_walkers',
            'pytest',
            ['training'],
            inputs=[future],
            )
    log0.result()

    #walkers = RandomWalker.multiply(7, dataset) + BiasedDynamicWalker.multiply(2, dataset[:5], bias=bias)
    #walkers[3].tag_unsafe()
    #walkers[7].tag_unsafe()
    #data_new = generate_all(walkers, model, reference, 1, 1)
    #data_new.length().result()
    #future = log_walkers(walkers)
    #log1 = to_wandb(
    #        'run_name',
    #        'test_log_dataset_walkers',
    #        'pytest',
    #        'dummy',
    #        ['walkers'],
    #        inputs=[future],
    #        )
    #log1.result()


def test_wandb_logger(context, dataset, mace_config, tmp_path):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
restraint: RESTRAINT ARG=CV AT=150 KAPPA=1
CV1: MATHEVAL ARG=CV VAR=a FUNC=2*a PERIODIC=NO
mtd: METAD ARG=CV1 PACE=1 SIGMA=10 HEIGHT=23
"""
    bias = PlumedBias(plumed_input)
    model = MACEModel(mace_config)
    model.initialize(dataset[:2])
    model.deploy()

    wandb_logger = WandBLogger(
            'pytest',
            'test_wandb_logger',
            error_x_axis='CV1',
            metric='mae',
            elements=['Cu', 'H'],
            indices=[0],
            )
    log = wandb_logger('4', model, data_train=bias.evaluate(dataset, as_dataset=True))
    log.result()


#def test_wandb_manual(context, dataset, mace_config, tmp_path):
#    plumed_input = """
#UNITS LENGTH=A ENERGY=kj/mol TIME=fs
#CV: VOLUME
#restraint: RESTRAINT ARG=CV AT=150 KAPPA=1
#CV1: MATHEVAL ARG=CV VAR=a FUNC=2*a PERIODIC=NO
#mtd: METAD ARG=CV1 PACE=1 SIGMA=10 HEIGHT=23
#"""
#    bias = PlumedBias(plumed_input)
#    model = MACEModel(mace_config)
#    model.initialize(dataset[:2])
#    model.deploy()
#
#    wandb.init(
#            name='0',
#            group='test_wandb_logger',
#            project='pytest',
#            resume='allow',
#            dir=tmp_path,
#            )
#    errors = Dataset.get_errors(
#            dataset,
#            model.evaluate(dataset),
#            ).result()
#    for i in range(dataset.length().result()):
#        kwargs = {
#                'index': i,
#                'energy': dataset[i].result().info['energy'],
#                'error_e': errors[i, 0],
#                'error_f': errors[i, 1],
#                }
#        wandb.log(kwargs)
#    wandb.finish()
