import pytest
import torch
import numpy as np
from parsl.app.futures import DataFuture

import psiflow
from psiflow.data import compute_rmse
from psiflow.hamiltonians import MACEHamiltonian
from psiflow.models import MACE
from psiflow.models.mace import KEY_ATOMIC_ENERGIES, KEY_ITERATION
from psiflow.utils.apps import copy_app_future
from psiflow.utils.io import _read_yaml

# tests for the calculator (specifying head, cueq, ..)?
# see test_mace_function..


def test_mace_init(tmp_path, mace_config, dataset):
    model = MACE.create(tmp_path / "mace", mace_config)
    atomic_energies = {"Cu": 3, "H": 7}
    for k, v in atomic_energies.items():
        model.add_atomic_energy(k, v)
    model.update_kwargs(seed=42, pair_repulsion=copy_app_future(True))

    assert model.model_future is None
    assert model.iteration == -1
    future = model.initialize(dataset[:5])
    assert isinstance(model.model_future, DataFuture)
    assert model.iteration == -1
    future.result()
    assert model.path_mlp.is_file()
    assert model.iteration == 0

    config = _read_yaml([model.path_config])
    assert config.pop(KEY_ATOMIC_ENERGIES) == atomic_energies
    assert config.pop(KEY_ITERATION) == 0
    assert config["seed"] == 42
    assert config["pair_repulsion"]

    mlp = torch.load(model.path_mlp, weights_only=False)
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
    training = dataset[:-5]
    validation = dataset[-5:]
    model = MACE.create(tmp_path / "mace", mace_config)

    model.train(training, validation)
    hamiltonian = model.create_hamiltonian()
    rmse0 = compute_rmse(
        validation.get(key),
        validation.evaluate(hamiltonian).get(key),
    )
    future_train = model.train(training, validation)
    hamiltonian = model.create_hamiltonian()
    rmse1 = compute_rmse(
        validation.get(key),
        validation.evaluate(hamiltonian).get(key),
    )

    future_train.result()  # wait for second training run
    model.lock.close()  # 'free' training dir
    del model

    # train from load
    model_ = MACE.load(tmp_path / "mace")
    hamiltonian = model_.create_hamiltonian()
    rmse2 = compute_rmse(
        validation.get(key),
        validation.evaluate(hamiltonian).get(key),
    )
    model_.train(training, validation)
    hamiltonian = model_.create_hamiltonian()
    rmse3 = compute_rmse(
        validation.get(key),
        validation.evaluate(hamiltonian).get(key),
    )

    print(rmse0.result())
    print(rmse1.result())
    print(rmse2.result())
    print(rmse3.result())

    assert rmse0.result() > rmse1.result()
    assert np.isclose(rmse1.result(), rmse2.result())
    assert rmse2.result() > rmse3.result()


def test_mace_hamiltonian(dataset, mace_foundation):
    hamiltonian0 = MACEHamiltonian(mace_foundation)
    hamiltonian1 = MACEHamiltonian(mace_foundation)

    assert hamiltonian0 == hamiltonian1
    hamiltonian1.update_kwargs(head="less")
    assert hamiltonian0 != hamiltonian1
    hamiltonian2 = psiflow.deserialize(psiflow.serialize(hamiltonian1)).result()
    assert hamiltonian0 != hamiltonian2

    e0 = hamiltonian0.compute(dataset, "energy")
    e1 = hamiltonian1.compute(dataset, "energy")
    assert np.allclose(e0.result(), e1.result())
    pass


