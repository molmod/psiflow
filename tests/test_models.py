import pytest
import os
import ast
import numpy as np
import torch
from dataclasses import asdict

from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

from ase.data import chemical_symbols
from ase.io.extxyz import read_extxyz

from psiflow.execution import ModelEvaluationExecution
from psiflow.data import Dataset
from psiflow.models import MACEModel, NequIPModel, AllegroModel, load_model


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
    for e in context[NequIPModel]:
        if type(e) == ModelEvaluationExecution:
            device = e.device
    torch.set_default_dtype(torch.float32)
    atoms = dataset[0].result().copy()
    atoms.calc = NequIPModel.load_calculator(
            path_model=model.deploy_future['float32'].result().filepath,
            device=device,
            dtype='float32',
            )
    assert atoms.calc.device == device
    e0 = atoms.get_potential_energy()

    torch.set_default_dtype(torch.float64)
    atoms = dataset[0].result().copy()
    atoms.calc = NequIPModel.load_calculator(
            path_model=model.deploy_future['float64'].result().filepath,
            device=device,
            dtype='float64',
            )
    assert atoms.calc.device == device
    e1 = atoms.get_potential_energy()

    assert np.allclose(e0, e1, atol=1e-4)
    assert not e0 == e1 # never exactly equal

    torch.set_default_dtype(torch.float32)
    e0 = model.evaluate(dataset.get(indices=[0]))[0].result().info['energy']
    model.reset()
    model.set_seed(1)
    model.initialize(dataset[:3])
    model.deploy()
    assert e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info['energy']
    model.reset()
    model.set_seed(0)
    model.initialize(dataset[:3])
    model.deploy()
    assert not e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info['energy']


def test_nequip_train(context, nequip_config, dataset, tmp_path):
    training   = dataset[:-5]
    validation = dataset[-5:]
    model = NequIPModel(context, nequip_config)
    model.initialize(training)
    model.deploy()
    errors0 = Dataset.get_errors(validation, model.evaluate(validation))
    model.train(training, validation)
    model.model_future.result()
    assert len(model.deploy_future) == 0
    model.deploy()
    errors1 = Dataset.get_errors(validation, model.evaluate(validation))
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
    e0 = model.evaluate(dataset.get(indices=[3]))[0].result().info['energy']

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
    e1 = model_.evaluate(dataset.get(indices=[3]))[0].result().info['energy']
    assert np.allclose(e0, e1, atol=1e-4) # up to single precision


@pytest.mark.skipif(torch.__version__.split('+')[0] != '1.11.0', reason='allegro only compatible with torch 1.11')
def test_allegro_init(context, allegro_config, dataset):
    model = AllegroModel(context, allegro_config)
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
    for e in context[AllegroModel]:
        if type(e) == ModelEvaluationExecution:
            device = e.device
    torch.set_default_dtype(torch.float32)
    atoms = dataset[0].result().copy()
    atoms.calc = AllegroModel.load_calculator(
            path_model=model.deploy_future['float32'].result().filepath,
            device=device,
            dtype='float32',
            )
    assert atoms.calc.device == device
    e0 = atoms.get_potential_energy()

    torch.set_default_dtype(torch.float64)
    atoms = dataset[0].result().copy()
    atoms.calc = AllegroModel.load_calculator(
            path_model=model.deploy_future['float64'].result().filepath,
            device=device,
            dtype='float64',
            )
    assert atoms.calc.device == device
    e1 = atoms.get_potential_energy()

    assert np.allclose(e0, e1, atol=1e-4)
    assert not e0 == e1 # never exactly equal

    torch.set_default_dtype(torch.float32)
    e0 = model.evaluate(dataset.get(indices=[0]))[0].result().info['energy']
    model.reset()
    model.set_seed(1)
    model.initialize(dataset[:3])
    model.deploy()
    assert e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info['energy']
    model.reset()
    model.set_seed(0)
    model.initialize(dataset[:3])
    model.deploy()
    assert not e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info['energy']


@pytest.mark.skipif(torch.__version__.split('+')[0] != '1.11.0', reason='allegro only compatible with torch 1.11')
def test_allegro_train(context, allegro_config, dataset, tmp_path):
    training   = dataset[:-5]
    validation = dataset[-5:]
    model = AllegroModel(context, allegro_config)
    model.initialize(training)
    model.deploy()
    errors0 = Dataset.get_errors(validation, model.evaluate(validation))
    model.train(training, validation)
    model.model_future.result()
    assert len(model.deploy_future) == 0
    model.deploy()
    errors1 = Dataset.get_errors(validation, model.evaluate(validation))
    assert np.mean(errors0.result(), axis=0)[1] > np.mean(errors1.result(), axis=0)[1]

    # test saving
    path_deployed = tmp_path / 'deployed.pth'
    model.save_deployed(path_deployed).result()
    assert os.path.isfile(path_deployed)


@pytest.mark.skipif(torch.__version__.split('+')[0] != '1.11.0', reason='allegro only compatible with torch 1.11')
def test_allegro_save_load(context, allegro_config, dataset, tmp_path):
    model = AllegroModel(context, allegro_config)
    future_raw, _, _ = model.save(tmp_path)
    assert future_raw.done()
    assert _ is None
    model.initialize(dataset[:2])
    model.deploy()
    e0 = model.evaluate(dataset.get(indices=[3]))[0].result().info['energy']

    path_config_raw = tmp_path / 'AllegroModel.yaml'
    path_config     = tmp_path / 'config_after_init.yaml'
    path_model      = tmp_path / 'model_undeployed.pth'
    futures = model.save(tmp_path)
    assert os.path.exists(path_config_raw)
    assert os.path.exists(path_config)
    assert os.path.exists(path_model)

    model_ = load_model(context, tmp_path)
    assert type(model_) == AllegroModel
    assert model_.model_future is not None
    model_.deploy()
    e1 = model_.evaluate(dataset.get(indices=[3]))[0].result().info['energy']
    assert np.allclose(e0, e1, atol=1e-4) # up to single precision


def test_mace_init_deploy(context, mace_config, dataset):
    model = MACEModel(context, mace_config)
    model.initialize(dataset[:1])
    initialized_config = model.config_future.result()
    assert initialized_config['avg_num_neighbors'] is not None
    e0s = ast.literal_eval(initialized_config['E0s'])
    assert len(e0s.keys()) == 2
    assert '1:' in initialized_config['E0s']
    assert '29:' in initialized_config['E0s']
    model.deploy()

    model = MACEModel(context, mace_config)
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
    for e in context[MACEModel]:
        if type(e) == ModelEvaluationExecution:
            device = e.device
    torch.set_default_dtype(torch.float32)
    atoms = dataset[0].result().copy()
    atoms.calc = MACEModel.load_calculator(
            path_model=model.deploy_future['float32'].result().filepath,
            device=device,
            dtype='float32',
            )
    assert atoms.calc.device.type == device
    e0 = atoms.get_potential_energy()

    torch.set_default_dtype(torch.float64)
    atoms = dataset[0].result().copy()
    atoms.calc = MACEModel.load_calculator(
            path_model=model.deploy_future['float64'].result().filepath,
            device=device,
            dtype='float64',
            )
    assert atoms.calc.device.type == device
    e1 = atoms.get_potential_energy()

    assert np.allclose(e0, e1, atol=1e-4)
    assert not e0 == e1 # never exactly equal

    torch.set_default_dtype(torch.float32)
    e0 = model.evaluate(dataset.get(indices=[0]))[0].result().info['energy']
    model.reset()
    model.set_seed(1)
    model.initialize(dataset[:3])
    model.deploy()
    assert e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info['energy']
    model.reset()
    model.set_seed(0)
    model.initialize(dataset[:3])
    model.deploy()
    assert not e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info['energy']


def test_mace_train(context, mace_config, dataset, tmp_path):
    # as an additional verification, this test can be executed while monitoring
    # the mace logging, and in particular the rmse_r during training, to compare
    # it with the manually computed value
    training   = dataset[:-5]
    validation = dataset[-5:]
    model = MACEModel(context, mace_config)
    model.initialize(training)
    model.deploy()
    errors0 = Dataset.get_errors(validation, model.evaluate(validation))
    model.train(training, validation)
    model.model_future.result()
    assert len(model.deploy_future) == 0
    model.deploy()
    errors1 = Dataset.get_errors(validation, model.evaluate(validation))
    assert np.mean(errors0.result(), axis=0)[1] > np.mean(errors1.result(), axis=0)[1]
    errors1 = Dataset.get_errors(validation, model.evaluate(validation))
    print('manual rmse_e, rmse_f: {}'.format(np.sqrt(np.mean(errors1.result() ** 2, axis=0)[:2])))

    # test saving
    path_deployed = tmp_path / 'deployed.pth'
    model.save_deployed(path_deployed).result()
    assert os.path.isfile(path_deployed)


def test_mace_save_load(context, mace_config, dataset, tmp_path):
    model = MACEModel(context, mace_config)
    future_raw, _, _ = model.save(tmp_path)
    assert future_raw.done()
    assert _ is None
    model.initialize(dataset[:2])
    model.deploy()
    e0 = model.evaluate(dataset.get(indices=[3]))[0].result().info['energy']

    path_config_raw = tmp_path / 'MACEModel.yaml'
    path_config     = tmp_path / 'config_after_init.yaml'
    path_model      = tmp_path / 'model_undeployed.pth'
    futures = model.save(tmp_path)
    assert os.path.exists(path_config_raw)
    assert os.path.exists(path_config)
    assert os.path.exists(path_model)

    model_ = load_model(context, tmp_path)
    assert type(model_) == MACEModel
    assert model_.model_future is not None
    model_.deploy()
    e1 = model_.evaluate(dataset.get(indices=[3]))[0].result().info['energy']
    assert np.allclose(e0, e1, atol=1e-4) # up to single precision
