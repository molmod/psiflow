import pytest
import logging
import os
from pathlib import Path
import wandb
import numpy as np

from psiflow.data import Dataset
from psiflow.models import NequIPModel
from psiflow.wandb_utils import WandBLogger, log_data, log_generators, to_wandb
from psiflow.reference import EMTReference
from psiflow.sampling import RandomWalker, DynamicWalker, PlumedBias
from psiflow.generator import Generator
from psiflow.checks import SafetyCheck, DiscrepancyCheck, \
        InteratomicDistanceCheck


@pytest.fixture
def generators(context, dataset):
    walker = RandomWalker(dataset[0])
    generators = Generator('random', walker).multiply(2)
    return generators


@pytest.fixture
def reference(context):
    return EMTReference()


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
    bias = PlumedBias(plumed_input)
    model = NequIPModel(nequip_config)
    model.initialize(dataset[:2])
    model.deploy()
    wandb_id = wandb.util.generate_id()
    future = log_data(
            dataset,
            bias=bias,
            model=model,
            error_kwargs=error_kwargs,
            )
    log0 = to_wandb(
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
            RandomWalker(dataset[0]),
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
    log1 = to_wandb(
            'run_name',
            'test_log_dataset_generators',
            'pytest',
            'dummy',
            ['generators'],
            inputs=[future],
            )
    log1.result()
