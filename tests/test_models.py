import pytest
import os
import numpy as np
import torch
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

from ase.data import chemical_symbols
from ase.io.extxyz import read_extxyz

from nequip.ase import NequIPCalculator

from psiflow.data import Dataset
from psiflow.models import NequIPModel, load_model
from psiflow.execution import ModelExecutionDefinition


def test_nequip_init(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config)
    model.set_seed(1)
    model.initialize(dataset[:3])
    assert isinstance(model.model_future, DataFuture)
    assert isinstance(model.config_future, AppFuture)
    assert len(model.deploy_future) == 0
    torch.load(model.model_future.result().filepath) # should work
    model.deploy()
    assert isinstance(model.deploy_future['float32'], DataFuture)
    assert isinstance(model.deploy_future['float64'], DataFuture)

    # simple test
    torch.set_default_dtype(torch.float32)
    atoms = dataset[0].result().copy()
    atoms.calc = NequIPModel.load_calculator(
            path_model=model.deploy_future['float32'].result().filepath,
            device=context[ModelExecutionDefinition].device,
            dtype='float32',
            )
    assert atoms.calc.device == context[ModelExecutionDefinition].device
    e0 = atoms.get_potential_energy()

    torch.set_default_dtype(torch.float64)
    atoms = dataset[0].result().copy()
    atoms.calc = NequIPModel.load_calculator(
            path_model=model.deploy_future['float64'].result().filepath,
            device=context[ModelExecutionDefinition].device,
            dtype='float64',
            )
    assert atoms.calc.device == context[ModelExecutionDefinition].device
    e1 = atoms.get_potential_energy()

    assert np.allclose(e0, e1, atol=1e-4)
    assert not e0 == e1 # never exactly equal

    torch.set_default_dtype(torch.float32)
    e0 = model.evaluate(dataset.get(indices=[0]))[0].result().info['energy_model']
    model.reset()
    model.set_seed(1)
    model.initialize(dataset[:3])
    model.deploy()
    assert e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info['energy_model']
    model.reset()
    model.set_seed(0)
    model.initialize(dataset[:3])
    model.deploy()
    assert not e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info['energy_model']


def test_nequip_train(context, nequip_config, dataset, tmp_path):
    training   = dataset[:-5]
    validation = dataset[-5:]
    model = NequIPModel(context, nequip_config)
    model.initialize(training)
    model.deploy()
    validation_evaluated = model.evaluate(validation, suffix='_my_model')
    assert 'energy_my_model' in validation_evaluated[0].result().info.keys()
    #errors0 = model.evaluate(validation, suffix='_model').get_errors()
    with pytest.raises(AssertionError):
        validation_evaluated.get_errors().result() # nonexisting suffix
    errors0 = validation_evaluated.get_errors(suffix_1='_my_model')
    epochs = model.train(training, validation).result()
    assert len(model.deploy_future) == 0
    assert epochs < nequip_config['max_epochs'] # terminates by app timeout
    assert epochs > 0
    model.deploy()
    errors1 = model.evaluate(validation).get_errors()
    evaluated = model.evaluate(validation)
    assert np.mean(errors0.result(), axis=0)[1] > np.mean(errors1.result(), axis=0)[1]

    # test saving
    path_deployed = tmp_path / 'deployed.pth'
    model.save_deployed(path_deployed).result()
    assert os.path.isfile(path_deployed)


def test_nequip_save_load(context, nequip_config, dataset, tmp_path):
    model = NequIPModel(context, nequip_config)
    future_raw, _, _ = model.save(tmp_path)
    assert future_raw.done()
    assert _ is None
    model.initialize(dataset[:2])
    model.deploy()
    e0 = model.evaluate(dataset.get(indices=[3]))[0].result().info['energy_model']

    path_config_raw = tmp_path / 'NequIPModel.yaml'
    path_config     = tmp_path / 'config_after_init.yaml'
    path_model      = tmp_path / 'model_undeployed.pth'
    futures = model.save(tmp_path)
    assert os.path.exists(path_config_raw)
    assert os.path.exists(path_config)
    assert os.path.exists(path_model)

    model_ = load_model(context, tmp_path)
    assert type(model_) == NequIPModel
    assert model_.model_future is not None
    model_.deploy()
    e1 = model_.evaluate(dataset.get(indices=[3]))[0].result().info['energy_model']
    assert np.allclose(e0, e1, atol=1e-4) # up to single precision
