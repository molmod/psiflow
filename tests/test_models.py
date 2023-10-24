import ast
import os

import numpy as np
import pytest
import torch
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.models import (
    AllegroModel,
    MACEConfig,
    MACEModel,
    NequIPConfig,
    NequIPModel,
    load_model,
)
from psiflow.reference import EMTReference


def test_nequip_init(nequip_config, dataset):
    model = NequIPModel(nequip_config)
    model.seed = 1
    model.initialize(dataset[:3])
    assert isinstance(model.model_future, DataFuture)
    assert isinstance(model.config_future, AppFuture)
    assert isinstance(model.deploy_future, DataFuture)
    torch.load(model.model_future.result().filepath)  # should work

    # simple test
    evaluation, _ = psiflow.context()[NequIPModel]
    device = "cuda" if evaluation.gpu else "cpu"
    torch.set_default_dtype(torch.float32)
    atoms = dataset[0].result().copy()
    atoms.calc = NequIPModel.load_calculator(
        path_model=model.deploy_future.result().filepath,
        device=device,
    )
    assert atoms.calc.device == device
    e0 = atoms.get_potential_energy()

    torch.set_default_dtype(torch.float32)
    e0 = model.evaluate(dataset.get(indices=[0]))[0].result().info["energy"]
    model.reset()
    model.seed = 1
    model.initialize(dataset[:3])
    assert e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info["energy"]
    model.reset()
    model.seed = 0
    model.initialize(dataset[:3])
    assert not e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info["energy"]


def test_nequip_train(gpu, nequip_config, dataset, tmp_path):
    training = dataset[:-5]
    validation = dataset[-5:]
    model = NequIPModel(nequip_config)
    model.initialize(training)
    with pytest.raises(AssertionError):
        model.add_atomic_energy("H", 1)  # cannot change this after initialization
    errors0 = Dataset.get_errors(validation, model.evaluate(validation))
    model.train(training, validation)
    model.model_future.result()
    errors1 = Dataset.get_errors(validation, model.evaluate(validation))
    assert np.mean(errors0.result(), axis=0)[1] > np.mean(errors1.result(), axis=0)[1]


def test_nequip_save_load(nequip_config, dataset, tmp_path):
    model = NequIPModel(nequip_config)
    future_raw, _, _ = model.save(tmp_path)
    assert not future_raw.done()
    assert _ is None
    model.initialize(dataset[:2])
    e0 = model.evaluate(dataset.get(indices=[3]))[0].result().info["energy"]

    path_config_raw = tmp_path / "NequIPModel.yaml"
    path_config = tmp_path / "config_after_init.yaml"
    path_model = tmp_path / "model_undeployed.pth"
    path_deploy = tmp_path / "model_deployed.pth"
    model.save(tmp_path, require_done=True)
    assert os.path.exists(path_config_raw)
    assert os.path.exists(path_config)
    assert os.path.exists(path_model)
    assert os.path.exists(path_deploy)

    model_ = load_model(tmp_path)
    assert type(model_) is NequIPModel
    assert model_.model_future is not None
    assert model_.deploy_future is not None
    e1 = model_.evaluate(dataset.get(indices=[3]))[0].result().info["energy"]
    assert np.allclose(e0, e1, atol=1e-4)  # up to single precision


def test_nequip_seed(nequip_config):
    config = NequIPConfig(**nequip_config)
    model = NequIPModel(config)
    assert model.seed == 123
    model.seed = 111
    assert model.seed == 111
    model.config_raw["seed"] = 112
    assert model.seed == 112


def test_nequip_offset(nequip_config, dataset):
    config = NequIPConfig(**nequip_config)
    model = NequIPModel(config)
    model.initialize(dataset[:2])
    with pytest.raises(AssertionError):
        model.add_atomic_energy("H", 1)  # cannot change this after initialization
    assert not model.do_offset
    errors = Dataset.get_errors(
        dataset,
        model.evaluate(dataset),
        properties=["energy"],
    )
    assert np.mean(errors.result()) < 1e3  # in meV/atom

    reference = EMTReference()
    atomic_energies = {
        "H": 3e2,
        "Cu": reference.compute_atomic_energy("Cu", box_size=6),  # future
    }
    errors_ = Dataset.get_errors(
        dataset.subtract_offset(**atomic_energies),
        model.evaluate(dataset),
    )
    assert np.mean(errors_.result()) > 1e3  # in meV/atom
    model.reset()
    for element, energy in atomic_energies.items():
        model.add_atomic_energy(element, energy)
    assert model.do_offset
    model.initialize(dataset[:2])
    errors_same = Dataset.get_errors(
        dataset,
        model.evaluate(dataset),
        properties=["energy"],
    )
    assert np.allclose(
        errors_same.result(),
        errors.result(),
        atol=1e-1,
    )


def test_allegro_init(allegro_config, dataset):
    model = AllegroModel(allegro_config)
    model.seed = 1
    model.initialize(dataset[:3])
    assert isinstance(model.model_future, DataFuture)
    assert isinstance(model.config_future, AppFuture)
    torch.load(model.model_future.result().filepath)  # should work

    # simple test
    evaluation, _ = psiflow.context()[AllegroModel]
    device = "cuda" if evaluation.gpu else "cpu"
    torch.set_default_dtype(torch.float32)
    atoms = dataset[0].result().copy()
    atoms.calc = AllegroModel.load_calculator(
        path_model=model.deploy_future.result().filepath,
        device=device,
    )
    assert atoms.calc.device == device
    e0 = atoms.get_potential_energy()

    e0 = model.evaluate(dataset.get(indices=[0]))[0].result().info["energy"]
    model.reset()
    model.seed = 1
    model.initialize(dataset[:3])
    assert e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info["energy"]
    model.reset()
    model.seed = 0
    model.initialize(dataset[:3])
    assert not e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info["energy"]


def test_allegro_train(gpu, allegro_config, dataset, tmp_path):
    training = dataset[:-5]
    validation = dataset[-5:]
    model = AllegroModel(allegro_config)
    model.initialize(training)
    errors0 = Dataset.get_errors(validation, model.evaluate(validation))
    model.train(training, validation)
    model.model_future.result()
    errors1 = Dataset.get_errors(validation, model.evaluate(validation))
    assert np.mean(errors0.result(), axis=0)[1] > np.mean(errors1.result(), axis=0)[1]


def test_allegro_save_load(allegro_config, dataset, tmp_path):
    model = AllegroModel(allegro_config)
    future_raw, _, _ = model.save(tmp_path)
    assert not future_raw.done()
    assert _ is None
    model.initialize(dataset[:2])
    e0 = model.evaluate(dataset.get(indices=[3]))[0].result().info["energy"]

    path_config_raw = tmp_path / "AllegroModel.yaml"
    path_config = tmp_path / "config_after_init.yaml"
    path_model = tmp_path / "model_undeployed.pth"
    path_deploy = tmp_path / "model_deployed.pth"
    model.save(tmp_path, require_done=True)
    assert os.path.exists(path_config_raw)
    assert os.path.exists(path_config)
    assert os.path.exists(path_model)
    assert os.path.exists(path_deploy)

    model_ = load_model(tmp_path)
    assert type(model_) is AllegroModel
    assert model_.model_future is not None
    assert model_.deploy_future is not None
    e1 = model_.evaluate(dataset.get(indices=[3]))[0].result().info["energy"]
    assert np.allclose(e0, e1, atol=1e-4)  # up to single precision


def test_mace_init(mace_config, dataset):
    model = MACEModel(mace_config)
    assert model.deploy_future is None
    assert model.model_future is None
    model.initialize(dataset[:1])
    assert model.deploy_future is not None
    assert model.model_future is not None
    initialized_config = model.config_future.result()
    assert initialized_config["avg_num_neighbors"] is not None
    e0s = ast.literal_eval(initialized_config["E0s"])
    assert len(e0s.keys()) == 2
    assert "1:" in initialized_config["E0s"]
    assert "29:" in initialized_config["E0s"]

    model = MACEModel(mace_config)
    model.seed = 1
    model.initialize(dataset[:3])
    assert isinstance(model.model_future, DataFuture)
    assert isinstance(model.config_future, AppFuture)
    torch.load(model.model_future.result().filepath)  # should work
    assert isinstance(model.deploy_future, DataFuture)

    # simple test
    evaluation, _ = psiflow.context()[MACEModel]
    device = "cuda" if evaluation.gpu else "cpu"
    torch.set_default_dtype(torch.float32)
    atoms = dataset[0].result().copy()
    atoms.calc = MACEModel.load_calculator(
        path_model=model.deploy_future.result().filepath,
        device=device,
    )
    assert atoms.calc.device.type == device
    e0 = atoms.get_potential_energy()

    e0 = model.evaluate(dataset.get(indices=[0]))[0].result().info["energy"]
    model.reset()
    model.seed = 1
    model.initialize(dataset[:3])
    assert e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info["energy"]
    model.reset()
    model.seed = 0
    model.initialize(dataset[:3])
    assert not e0 == model.evaluate(dataset.get(indices=[0]))[0].result().info["energy"]


def test_mace_train(gpu, mace_config, dataset, tmp_path):
    # as an additional verification, this test can be executed while monitoring
    # the mace logging, and in particular the rmse_r during training, to compare
    # it with the manually computed value
    training = dataset[:-5]
    validation = dataset[-5:]
    model = MACEModel(mace_config)
    model.initialize(training)
    errors0 = Dataset.get_errors(validation, model.evaluate(validation))
    model.train(training, validation)
    model.model_future.result()
    model.reset()
    model.initialize(training)
    model.train(training, validation)
    errors1 = Dataset.get_errors(validation, model.evaluate(validation))
    assert np.mean(errors0.result(), axis=0)[1] > np.mean(errors1.result(), axis=0)[1]
    errors1 = Dataset.get_errors(validation, model.evaluate(validation))
    print(
        "manual rmse_e, rmse_f: {}".format(
            np.sqrt(np.mean(errors1.result() ** 2, axis=0)[:2])
        )
    )


def test_mace_save_load(mace_config, dataset, tmp_path):
    model = MACEModel(mace_config)
    model.add_atomic_energy("H", 3)
    model.add_atomic_energy("Cu", 4)
    future_raw, _, _ = model.save(tmp_path)
    assert not future_raw.done()  # do not wait for result by default
    assert _ is None
    model.initialize(dataset[:2])
    e0 = model.evaluate(dataset.get(indices=[3]))[0].result().info["energy"]

    path_config_raw = tmp_path / "MACEModel.yaml"
    path_config = tmp_path / "config_after_init.yaml"
    path_model = tmp_path / "model_undeployed.pth"
    path_deployed = tmp_path / "model_deployed.pth"
    model.save(tmp_path, require_done=True)
    assert os.path.exists(path_config_raw)
    assert os.path.exists(path_config)
    assert os.path.exists(path_model)
    assert os.path.exists(path_deployed)

    model_ = load_model(tmp_path)
    assert type(model_) is MACEModel
    assert model_.model_future is not None
    assert model_.deploy_future is not None
    e1 = model_.evaluate(dataset.get(indices=[3]))[0].result().info["energy"]
    assert np.allclose(e0, e1, atol=1e-4)  # up to single precision


def test_mace_seed(mace_config):
    config = MACEConfig(**mace_config)
    model = MACEModel(config)
    assert model.seed == 0
    model.seed = 111
    assert model.seed == 111
    model.config_raw["seed"] = 112
    assert model.seed == 112


def test_mace_offset(mace_config, dataset, tmp_path):
    config = MACEConfig(**mace_config)
    model = MACEModel(config)
    model.initialize(dataset[:2])
    with pytest.raises(AssertionError):
        model.add_atomic_energy("H", 1)  # cannot change this after initialization
    assert not model.do_offset
    errors = Dataset.get_errors(
        dataset,
        model.evaluate(dataset),
        properties=["energy"],
    )
    assert np.mean(errors.result()) < 1e3  # in meV/atom

    reference = EMTReference()
    atomic_energies = {
        "H": 3e2,
        "Cu": reference.compute_atomic_energy("Cu", box_size=6),  # future
    }
    errors_ = Dataset.get_errors(
        dataset.subtract_offset(**atomic_energies),
        model.evaluate(dataset),
    )
    assert np.mean(errors_.result()) > 1e3  # in meV/atom
    model.reset()
    for element, energy in atomic_energies.items():
        model.add_atomic_energy(element, energy)
    assert model.do_offset
    model.initialize(dataset[:2])
    errors_same = Dataset.get_errors(
        dataset,
        model.evaluate(dataset),
        properties=["energy"],
    )
    assert np.allclose(
        errors_same.result(),
        errors.result(),
        atol=1e-1,
    )
    model.save(tmp_path)
    psiflow.wait()
    model = load_model(tmp_path)
    assert model.atomic_energies["H"] == atomic_energies["H"]
    assert model.atomic_energies["Cu"] == atomic_energies["Cu"].result()


def test_model_evaluate(mace_config, dataset):
    model = MACEModel(mace_config)
    model.initialize(dataset[:1])

    reference = model.evaluate(dataset, batch_size=10000)
    evaluated = model.evaluate(dataset, batch_size=3)
    evaluated_ = model.evaluate(dataset, batch_size=4)
    evaluated__ = model.evaluate(dataset, batch_size=1)
    assert evaluated.length().result() == dataset.length().result()
    assert evaluated_.length().result() == dataset.length().result()
    assert evaluated__.length().result() == dataset.length().result()
    for i in range(dataset.length().result()):
        assert np.allclose(
            evaluated[i].result().info["energy"],
            reference[i].result().info["energy"],
        )
        assert np.allclose(
            evaluated[i].result().info["energy"],
            evaluated_[i].result().info["energy"],
        )
        assert np.allclose(
            evaluated[i].result().info["energy"],
            evaluated__[i].result().info["energy"],
        )
