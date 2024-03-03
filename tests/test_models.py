import copy

import numpy as np
import torch
from parsl.app.futures import DataFuture

import psiflow
from psiflow.data import Dataset
from psiflow.models import MACEModel, load_model


def test_mace_init(mace_config, dataset):
    model = MACEModel(**mace_config)
    assert model.model_future is None
    model.initialize(dataset[:1])
    assert model.model_future is not None

    config = copy.deepcopy(mace_config)
    config[
        "batch_size"
    ] = 100000  # bigger than ntrain --> should get reduced internally
    model = MACEModel(**config)
    model.seed = 1
    model.initialize(dataset[:3])
    assert isinstance(model.model_future, DataFuture)
    torch.load(model.model_future.result().filepath)  # should work

    # create hamiltonian and verify addition of atomic energies
    hamiltonian = model.create_hamiltonian()
    assert hamiltonian == model.create_hamiltonian()
    evaluated = hamiltonian.evaluate(dataset)

    nstates = dataset.length().result()
    energies = np.array([evaluated[i].result().info["energy"] for i in range(nstates)])
    assert not np.any(np.allclose(energies, 0.0))
    energy_Cu = 3
    energy_H = 7
    hamiltonian.atomic_energies = {
        "Cu": energy_Cu,
        "H": energy_H,
    }
    assert hamiltonian != model.create_hamiltonian()  # atomic energies

    evaluated_ = hamiltonian.evaluate(dataset)
    for i in range(nstates):
        assert np.allclose(
            energies[i],
            evaluated_.subtract_offset(Cu=energy_Cu, H=energy_H)[i]
            .result()
            .info["energy"],
        )

    hamiltonian.atomic_energies = {"Cu": 0, "H": 0, "jasldfkjsadf": 0}
    evaluated__ = hamiltonian.evaluate(dataset)
    for i in range(nstates):
        assert np.allclose(
            energies[i],
            evaluated__[i].result().info["energy"],
        )
    hamiltonian = model.create_hamiltonian()
    model.reset()
    model.initialize(dataset[:3])
    assert hamiltonian != model.create_hamiltonian()


def test_mace_train(gpu, mace_config, dataset, tmp_path):
    # as an additional verification, this test can be executed while monitoring
    # the mace logging, and in particular the rmse_r during training, to compare
    # it with the manually computed value
    training = dataset[:-5]
    validation = dataset[-5:]
    model = MACEModel(**mace_config)
    model.initialize(training)
    hamiltonian0 = model.create_hamiltonian()
    errors0 = Dataset.get_errors(validation, hamiltonian0.evaluate(validation))
    model.train(training, validation)
    hamiltonian1 = model.create_hamiltonian()
    errors1 = Dataset.get_errors(validation, hamiltonian1.evaluate(validation))
    assert np.mean(errors0.result(), axis=0)[1] > np.mean(errors1.result(), axis=0)[1]


def test_mace_save_load(mace_config, dataset, tmp_path):
    model = MACEModel(**mace_config)
    model.add_atomic_energy("H", 3)
    model.add_atomic_energy("Cu", 4)
    model.save(tmp_path)
    model.initialize(dataset[:2])
    e0 = (
        model.create_hamiltonian()
        .evaluate(dataset.get(indices=[3]))[0]
        .result()
        .info["energy"]
    )

    psiflow.wait()
    assert (tmp_path / "MACEModel.yaml").exists()
    assert not (tmp_path / "MACEModel.pth").exists()

    model.save(tmp_path)
    psiflow.wait()
    assert (tmp_path / "MACEModel.pth").exists()

    model_ = load_model(tmp_path)
    assert type(model_) is MACEModel
    assert model_.model_future is not None
    e1 = (
        model_.create_hamiltonian()
        .evaluate(dataset.get(indices=[3]))[0]
        .result()
        .info["energy"]
    )
    assert np.allclose(e0, e1, atol=1e-4)  # up to single precision


def test_mace_seed(mace_config):
    model = MACEModel(**mace_config)
    assert model.seed == 0
    model.seed = 111
    assert model.seed == 111
    model.config.seed = 112
    assert model.seed == 112
