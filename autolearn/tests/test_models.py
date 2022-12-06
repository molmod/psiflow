import pytest
import numpy as np
import torch
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

from ase.data import chemical_symbols
from ase.io.extxyz import read_extxyz

from nequip.ase import NequIPCalculator

from autolearn import Dataset
from autolearn.models import NequIPModel
from autolearn.execution import ModelExecutionDefinition

from common import context, nequip_config
from test_dataset import dataset


def test_nequip_init_deploy_evaluate(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config)
    model.initialize(dataset)
    assert isinstance(model.future, DataFuture)
    assert isinstance(model.future_config, AppFuture)
    torch.load(model.future.result().filepath) # should work
    model.deploy()

    # simple test
    calculator = model.load_calculator(
            path_model=model.future_deploy.result().filepath,
            device=context[ModelExecutionDefinition].device,
            dtype=context[ModelExecutionDefinition].dtype,
            )
    assert calculator.device == context[ModelExecutionDefinition].device
    model.evaluate(dataset) # overwrites dataset
    with open(dataset.future.result(), 'r') as f:
        data_evaluated = list(read_extxyz(f, index=slice(None)))
    for atoms in data_evaluated:
        assert 'energy_model' in atoms.info.keys()
        assert 'stress_model' in atoms.info.keys()
        assert 'forces_model' in atoms.arrays.keys()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires GPU')
def test_nequip_train(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config)
    training   = dataset[:-10]
    validation = dataset[-10:]
    model.initialize(training)
    model.deploy()
    model.train(training, validation)
    model.deploy()
    model.evaluate(validation)
    # ensure everything is executed before closing context
    validation.length().result()
