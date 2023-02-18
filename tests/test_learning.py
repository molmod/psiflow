import shutil
import pytest

from psiflow.learning import BatchLearning, load_learning
from psiflow.wandb_utils import WandBLogger


def test_learning_save_load(context, tmp_path):
    path_output = tmp_path / 'output'
    path_output.mkdir()
    learning = BatchLearning(
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
    learning = BatchLearning(
            path_output=path_output,
            wandb_logger=wandb_logger,
            pretraining_nstates=99,
            )
    learning_ = load_learning(path_output)
    assert learning_.pretraining_nstates == 99
    assert learning_.wandb_logger is not None
