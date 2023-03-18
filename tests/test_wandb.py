import pytest
import logging
import os
from pathlib import Path
import numpy as np

from psiflow.data import Dataset
from psiflow.generate import generate_all
from psiflow.models import NequIPModel
from psiflow.wandb_utils import WandBLogger, log_data, log_walkers, to_wandb
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
            dataset,
            bias=bias,
            model=model,
            error_kwargs=error_kwargs,
            )
    log0 = to_wandb(
            'run_name',
            'test_log_dataset_walkers',
            'pytest',
            'CV',
            ['training'],
            inputs=[future],
            )
    log0.result()

    walkers = RandomWalker.multiply(7, dataset) + BiasedDynamicWalker.multiply(2, dataset[:5], bias=bias)
    walkers[3].tag_unsafe()
    walkers[7].tag_unsafe()
    data_new = generate_all(walkers, model, reference, 1, 1)
    data_new.length().result()
    future = log_walkers(walkers)
    log1 = to_wandb(
            'run_name',
            'test_log_dataset_walkers',
            'pytest',
            'dummy',
            ['walkers'],
            inputs=[future],
            )
    log1.result()
