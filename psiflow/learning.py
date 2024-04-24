from pathlib import Path
from typing import Optional, Union

import typeguard
from parsl.app.app import join_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.models import Model
from psiflow.reference import Reference
from psiflow.sampling import SimulationOutput, Walker


@join_app
def evaluate(
    outputs: list[SimulationOutput],
    reference: Reference,
    identifier: int,
    data: Dataset,  # stores evaluated states
    error_thresholds_for_reset: tuple[float, float],
    error_thresholds_for_discard: tuple[float, float],
    *output_lengths: int,
) -> AppFuture:
    pass


@typeguard.typechecked
@psiflow.serializable
class Learning:
    model: Model
    reference: Reference
    path_output: Path
    identifier: Union[AppFuture, int]
    train_valid_split: float
    mix_training_validation: bool
    error_thresholds_for_reset: tuple[float, float]
    error_thresholds_for_discard: tuple[float, float]
    wandb_group: Optional[str]
    wandb_name: Optional[str]
    iteration: int

    def __init__(
        self,
        model: Model,
        reference: Reference,
        path_output: Union[str, Path],
        identifier: int = 0,
        train_valid_split: float = 0.9,
        mix_train_valid: bool = True,
        error_thresholds_for_reset: tuple[float, float] = (20, 300),
        error_thresholds_for_discard: tuple[float, float] = (30, 600),
        wandb_group: Optional[str] = None,
        wandb_name: Optional[str] = None,
    ):
        self.model = model
        self.reference = reference
        self.path_output = Path(path_output)
        self.path_output.mkdir(exist_ok=False, parents=True)

        self.identifier = identifier
        self.train_valid_split = train_valid_split
        self.mix_train_valid = mix_train_valid
        self.error_thresholds_for_reset = error_thresholds_for_reset
        self.error_thresholds_for_discard = error_thresholds_for_discard
        self.wandb_group = wandb_group
        self.wandb_name = wandb_name

        self.iteration = 0

    def skip(self) -> bool:
        pass

    def save(
        self, serialize: bool = True
    ) -> bool:  # human-readable format and serialized
        assert not self.skip()

    def pretraining(
        self,
        walkers: list[Walker],
        universal_potential: str,
        **sampling_kwargs,
    ) -> Dataset:
        pass

    def sample_qm_train(
        self,
        walkers: list[Walker],
        **sampling_kwargs,
    ) -> Dataset:
        pass
