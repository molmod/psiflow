import shutil
import pytest

from psiflow.reference import EMTReference
from psiflow.sampling import RandomWalker
from psiflow.models import MACEModel
from psiflow.learning import SequentialLearning, ConcurrentLearning, \
        load_learning
from psiflow.wandb_utils import WandBLogger


def test_learning_save_load(context, tmp_path):
    path_output = tmp_path / 'output'
    path_output.mkdir()
    learning = SequentialLearning(
            path_output=path_output,
            wandb_logger=None,
            pretraining_nstates=100,
            )
    learning_ = load_learning(path_output)
    assert learning_.pretraining_nstates == 100

    shutil.rmtree(path_output)
    path_output.mkdir()
    wandb_logger = WandBLogger(
            wandb_project='pytest',
            wandb_group='test_learning_save_load',
            )
    learning = SequentialLearning(
            path_output=path_output,
            wandb_logger=wandb_logger,
            pretraining_nstates=99,
            )
    learning_ = load_learning(path_output)
    assert learning_.pretraining_nstates == 99
    assert learning_.wandb_logger is not None


def test_sequential_learning(context, tmp_path, mace_config, dataset):
    path_output = tmp_path / 'output'
    path_output.mkdir()
    reference = EMTReference()

    learning = SequentialLearning(
            path_output=path_output,
            wandb_logger=None,
            pretraining_nstates=50,
            train_from_scratch=True,
            use_formation_energy=True,
            train_valid_split=0.8,
            niterations=1,
            )
    model = MACEModel(mace_config)
    model.initialize(dataset[:2])

    walkers = RandomWalker.multiply(
            5,
            dataset,
            amplitude_pos=0.05,
            amplitude_box=0,
            )
    data_train, data_valid = learning.run(model, reference, walkers, dataset)
    assert data_train.length().result() == 20
    assert data_valid.length().result() == 5

    data_train, data_valid = learning.run(model, reference, walkers)
    assert data_train.length().result() == 0 # iteration 0 already performed
    assert data_valid.length().result() == 0

    model.reset()
    data_train, data_valid = learning.run(model, reference, walkers)
    assert data_train.length().result() == 40 # because of pretraining
    assert data_valid.length().result() == 10


def test_concurrent_learning(context, tmp_path, mace_config, dataset):
    path_output = tmp_path / 'output'
    path_output.mkdir()
    reference = EMTReference()

    learning = ConcurrentLearning(
            path_output=path_output,
            wandb_logger=None,
            pretraining_nstates=50,
            train_from_scratch=True,
            use_formation_energy=True,
            train_valid_split=0.8,
            niterations=2,
            min_states_per_iteration=8,
            max_states_per_iteration=40,
            )
    model = MACEModel(mace_config)
    model.initialize(dataset[:2])

    walkers = RandomWalker.multiply(
            10,
            dataset,
            amplitude_pos=0.05,
            amplitude_box=0,
            )
    data_train, data_valid = learning.run(model, reference, walkers, dataset)
    assert data_train.length().result() == 16 + 2 * 32
    assert data_valid.length().result() == 4  + 2 * 8
