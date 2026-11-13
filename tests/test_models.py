import pytest
import torch
import numpy as np
from parsl.app.futures import DataFuture

import yaml
import psiflow
from psiflow.compute import compare_arrays
from psiflow.hamiltonians import MACEHamiltonian
from psiflow.models import MACE
from psiflow.models.mace import KEY_ATOMIC_ENERGIES, KEY_ITERATION, MODEL_DIRS
from psiflow.utils.apps import copy_app_future


def test_mace_init(tmp_path, mace_config, dataset):
    model = MACE.create(tmp_path / "mace", mace_config)
    atomic_energies = {"Cu": 3, "H": 7}
    for k, v in atomic_energies.items():
        model.add_atomic_energy(k, v)
    model.update_kwargs(seed=42, pair_repulsion=copy_app_future(True))

    assert model.model_future is None
    assert model.iteration == -1
    model.initialize(dataset[:5])
    assert isinstance(model.model_future, DataFuture)
    assert model.iteration == 0
    model.wait_for.result()

    config = yaml.safe_load(model.path_config.open())
    assert config.pop(KEY_ATOMIC_ENERGIES) == atomic_energies
    assert config.pop(KEY_ITERATION) == 0
    assert config["seed"] == 42
    assert config["pair_repulsion"]

    mlp = torch.load(model._get_final_model(), weights_only=False)
    atomic_energies_ = mlp.atomic_energies_fn.atomic_energies.numpy()
    assert atomic_energies_.flatten().tolist() == [7, 3]  # H, Cu

    with pytest.raises(AssertionError):  # can only initialise once
        model.initialize(dataset[:3])
    with pytest.raises(AssertionError):  # one instance per dir
        MACE.load(tmp_path / "mace")

    model.config = model.atomic_energies = None
    model._load_config()
    assert model.config == config
    assert model.atomic_energies == atomic_energies
    assert model.iteration == 0


def test_mace_train(gpu, mace_config, dataset, tmp_path):
    # as an additional verification, this test can be executed while monitoring
    # the mace logging, and in particular the rmse_r during training, to compare
    # it with the manually computed value
    key = "per_atom_energy"
    training = dataset[:-15]
    validation = dataset[-5:]
    path = tmp_path / "mace"
    model = MACE.create(path, mace_config)
    assert model.iteration == -1
    [data] = validation.get(key)

    model.train(training, validation)
    assert model.iteration == 1  # init + train
    hamiltonian = model.create_hamiltonian()
    validation0 = hamiltonian.evaluate(validation)
    [data0] = validation0.get(key)
    future_train = model.train(training, validation)
    assert model.iteration == 2
    hamiltonian = model.create_hamiltonian()
    validation1 = hamiltonian.evaluate(validation)
    [data1] = validation1.get(key)

    future_train.result()  # wait for second training run
    with pytest.raises(AssertionError):
        MACE.load(path)
    MODEL_DIRS.pop(str(model.root))  # 'free' training dir

    # train from load
    model_ = MACE.load(path)
    assert model_.iteration == 2
    hamiltonian = model_.create_hamiltonian()
    validation2 = hamiltonian.evaluate(validation)
    [data2] = validation2.get(key)
    model_.train(training, validation)
    assert model_.iteration == 3
    hamiltonian = model_.create_hamiltonian()
    validation3 = hamiltonian.evaluate(validation)
    [data3] = validation3.get(key)

    rmse0 = compare_arrays(data, data0).result()
    rmse1 = compare_arrays(data, data1).result()
    rmse2 = compare_arrays(data, data2).result()
    rmse3 = compare_arrays(data, data3).result()

    print(rmse0)
    print(rmse1)
    print(rmse2)
    print(rmse3)
    assert rmse0 > rmse1
    assert np.isclose(rmse1, rmse2)
    assert rmse2 > rmse3


def test_mace_reset(gpu, mace_config, dataset, tmp_path):
    model = MACE.create(tmp_path / "mace", mace_config)
    model.update_kwargs(max_num_epochs=1)
    training = dataset[:2]
    validation = dataset[-1:]

    # 0-init and 1-train
    model.train(training, validation)
    hamiltonian = model.create_hamiltonian()
    out0 = hamiltonian.evaluate(validation)

    model.reset()

    # 2-init and 3-train
    model.train(training, validation)
    hamiltonian = model.create_hamiltonian()
    out1 = hamiltonian.evaluate(validation)

    # 4-train
    model.train(training, validation)

    # verify all tasks finish correctly
    out0.reset()
    out1.reset()
    model.wait_for.result()

    assert list(model.path_checkpoints.glob("0-init*.model"))
    assert list(model.path_checkpoints.glob("1-train*.model"))
    assert list(model.path_checkpoints.glob("2-init*.model"))
    assert list(model.path_checkpoints.glob("3-train*.model"))
    assert list(model.path_checkpoints.glob("4-train*.model"))


def test_mace_hamiltonian(dataset, mace_foundation):
    hamiltonian0 = MACEHamiltonian(mace_foundation)
    hamiltonian1 = MACEHamiltonian(mace_foundation)

    assert hamiltonian0 == hamiltonian1
    hamiltonian1.update_kwargs(head="less")
    assert hamiltonian0 != hamiltonian1
    hamiltonian2 = psiflow.deserialize(psiflow.serialize(hamiltonian1)).result()
    assert hamiltonian0 != hamiltonian2
    assert hamiltonian1 == hamiltonian2
    hamiltonian2.update_kwargs(enable_cueq=True)

    out0 = hamiltonian0.compute(dataset)
    out1 = hamiltonian1.compute(dataset)
    out2 = hamiltonian2.compute(dataset)
    out0, out1, out2 = out0.result(), out1.result(), out2.result()
    for key in out0.keys:
        # single precision
        assert np.allclose(out0.get(key), out1.get(key), atol=1e-6)
        assert np.allclose(out0.get(key), out2.get(key), atol=1e-6)
