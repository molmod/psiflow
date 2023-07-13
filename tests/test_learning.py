import shutil
import pytest

from psiflow.reference import EMTReference
from psiflow.sampling import RandomWalker, PlumedBias, BiasedDynamicWalker
from psiflow.models import MACEModel
from psiflow.learning import SequentialLearning, load_learning
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
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
"""
    bias = PlumedBias(plumed_input)

    wandb_logger = WandBLogger('pytest', 'test_sequential_index', error_x_axis='CV')

    learning = SequentialLearning(
            path_output=path_output,
            wandb_logger=wandb_logger,
            pretraining_nstates=50,
            train_from_scratch=True,
            use_formation_energy=False,
            train_valid_split=0.8,
            niterations=1,
            )
    model = MACEModel(mace_config)
    model.config_raw['max_num_epochs'] = 1
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
    wandb_logger = WandBLogger('pytest', 'test_sequential_cv', error_x_axis='CV')
    path_output = tmp_path / 'output_'
    path_output.mkdir()
    learning = SequentialLearning(
            path_output=path_output,
            wandb_logger=wandb_logger,
            pretraining_nstates=50,
            train_from_scratch=True,
            use_formation_energy=True,
            train_valid_split=0.8,
            niterations=1,
            )
    walkers = BiasedDynamicWalker.multiply(
            5,
            dataset,
            bias=bias,
            steps=10,
            step=1,
            )
    data_train, data_valid = learning.run(model, reference, walkers)
    assert model.use_formation_energy
    assert 'formation_energy' in data_train.energy_labels().result()
    assert 'formation_energy' in model.evaluate(data_train).energy_labels().result()
    assert (path_output / 'random_pretraining').is_dir()
    assert data_train.length().result() == 44 # because of pretraining
    assert data_valid.length().result() == 11
