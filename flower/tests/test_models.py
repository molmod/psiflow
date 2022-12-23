import pytest
import os
import numpy as np
import torch
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

from ase.data import chemical_symbols
from ase.io.extxyz import read_extxyz

from nequip.ase import NequIPCalculator

from flower.data import Dataset
from flower.models import NequIPModel
from flower.execution import ModelExecutionDefinition

from common import context, nequip_config
from test_dataset import dataset


def test_nequip_init(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config, dataset[:3])
    assert isinstance(model.model_future, DataFuture)
    assert isinstance(model.config_future, AppFuture)
    assert len(model.deploy_future) == 0
    torch.load(model.model_future.result().filepath) # should work
    model.deploy()
    assert isinstance(model.deploy_future['float32'], DataFuture)
    assert isinstance(model.deploy_future['float64'], DataFuture)

    # simple test
    atoms = dataset[0].result()
    atoms.calc = calculator = NequIPModel.load_calculator(
            path_model=model.deploy_future['float32'].result().filepath,
            device=context[ModelExecutionDefinition].device,
            dtype=context[ModelExecutionDefinition].dtype,
            )
    assert atoms.calc.device == context[ModelExecutionDefinition].device
    e0 = atoms.get_potential_energy()
    torch.set_default_dtype(torch.float64)
    atoms.calc = calculator = NequIPModel.load_calculator(
            path_model=model.deploy_future['float64'].result().filepath,
            device=context[ModelExecutionDefinition].device,
            dtype=context[ModelExecutionDefinition].dtype,
            )
    assert atoms.calc.device == context[ModelExecutionDefinition].device
    e1 = atoms.get_potential_energy()
    assert np.allclose(e0, e1, atol=1e-4)
    assert not e0 == e1 # never exactly equal


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires GPU')
def test_nequip_train(context, nequip_config, dataset, tmp_path):
    training   = dataset[:-5]
    validation = dataset[-5:]
    model = NequIPModel(context, nequip_config, training)
    model.deploy()
    validation_evaluated = model.evaluate(validation, suffix='_my_model')
    assert 'energy_my_model' in validation_evaluated[0].result().info.keys()
    #errors0 = model.evaluate(validation, suffix='_model').get_errors()
    with pytest.raises(AssertionError):
        validation_evaluated.get_errors().result() # nonexisting suffix
    errors0 = validation_evaluated.get_errors(suffix_1='_my_model')
    model.train(training, validation)
    assert len(model.deploy_future) == 0
    model.deploy()
    errors1 = model.evaluate(validation).get_errors()
    evaluated = model.evaluate(validation)
    assert np.mean(errors0.result(), axis=0)[1] > np.mean(errors1.result(), axis=0)[1]

    # test saving
    path_deployed = tmp_path / 'deployed.pth'
    model.save_deployed(path_deployed).result()
    assert os.path.isfile(path_deployed)
