from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, assign_identifier
from psiflow.geometry import Geometry, NullState
from psiflow.metrics import Metrics
from psiflow.models import Model
from psiflow.reference import Reference
from psiflow.sampling import SimulationOutput, Walker
from psiflow.utils import unpack_i


def _compute_error(
    state0: Geometry,
    state1: Geometry,
) -> tuple[float, float]:
    if state0 == NullState or state1 == NullState:
        return np.nan, np.nan
    elif state0.energy is None or state1.energy is None:
        return np.nan, np.nan
    else:
        e_rmse = np.abs(state0.energy - state1.energy)
        f_rmse = np.sqrt(np.mean((state0.forces - state1.forces) ** 2))
        return e_rmse, f_rmse


compute_error = python_app(_compute_error, executors=["default_threads"])


def _exceeds_error(
    errors: tuple[float, float],
    thresholds: tuple[float, float],
) -> bool:
    if np.isnan(errors[0]) and not np.isnan(thresholds[0]):
        return True
    if np.isnan(errors[1]) and not np.isnan(thresholds[1]):
        return True
    return (errors[0] > thresholds[0]) or (errors[1] > thresholds[1])


exceeds_error = python_app(_exceeds_error, executors=["default_threads"])


def evaluate_outputs(
    outputs: list[SimulationOutput],
    model: Model,
    reference: Reference,
    identifier: Union[AppFuture, int],
    error_thresholds_for_reset: tuple[float, float],
    error_thresholds_for_discard: tuple[float, float],
    metrics: Optional[Metrics] = None,
) -> Dataset:
    states = [o.state for o in outputs]
    eval_ref = [reference.evaluate(s) for s in states]
    eval_mod = model.evaluate(Dataset(states))
    errors = [compute_error(s, eval_mod[i]) for i, s in enumerate(eval_ref)]
    processed_states = []
    resets = []
    for i, state in enumerate(eval_ref):
        discard = exceeds_error(errors[i], error_thresholds_for_discard)
        reset = exceeds_error(errors[i], error_thresholds_for_reset)
        resets.append(reset)

        _ = assign_identifier(state, discard, identifier)
        assigned = unpack_i(_, 0)
        identifier = unpack_i(_, 1)
        processed_states.append(assigned)

    data = Dataset(processed_states)
    return identifier, data, resets


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
        initial_data: Optional[Dataset] = None,
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
        if initial_data is None:
            self.data = Dataset([])
        else:
            self.data = initial_data

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
