import pytest
import os
import numpy as np
import torch
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

from ase.data import chemical_symbols
from ase.io.extxyz import read_extxyz

from nequip.ase import NequIPCalculator

from flower import Dataset
from flower.models import NequIPModel
from flower.execution import ModelExecutionDefinition

from common import context, nequip_config
from test_dataset import dataset


def test_nequip_init(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config, dataset[:3])
    assert isinstance(model.model_future, DataFuture)
    assert isinstance(model.config_future, AppFuture)
    assert model.deploy_future is None
    torch.load(model.model_future.result().filepath) # should work
    model.deploy()
    assert isinstance(model.deploy_future, DataFuture)

    # simple test
    calculator = NequIPModel.load_calculator(
            path_model=model.deploy_future.result().filepath,
            device=context[ModelExecutionDefinition].device,
            dtype=context[ModelExecutionDefinition].dtype,
            )
    assert calculator.device == context[ModelExecutionDefinition].device


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires GPU')
def test_nequip_train(context, nequip_config, dataset, tmp_path):
    training   = dataset[:-5]
    validation = dataset[-5:]
    model = NequIPModel(context, nequip_config, training)
    model.deploy()
    errors0 = model.evaluate(validation).get_errors()
    model.train(training, validation)
    assert model.deploy_future is None
    model.deploy()
    errors1 = model.evaluate(validation).get_errors()
    evaluated = model.evaluate(validation)
    assert np.mean(errors0.result(), axis=0)[1] > np.mean(errors1.result(), axis=0)[1]

    # test saving
    path_deployed = tmp_path / 'deployed.pth'
    model.save_deployed(path_deployed).result()
    assert os.path.isfile(path_deployed)
