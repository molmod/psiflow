from __future__ import annotations  # necessary for type-guarding class methods

import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry, NullState, assign_identifier
from psiflow.hamiltonians import Hamiltonian, Zero
from psiflow.metrics import Metrics
from psiflow.models import Model
from psiflow.reference import Reference, evaluate
from psiflow.sampling import SimulationOutput, Walker, sample
from psiflow.utils.apps import boolean_or, isnan, setup_logger, unpack_i

logger = setup_logger(__name__)


@typeguard.typechecked
def _compute_error(
    state0: Geometry,
    state1: Geometry,
) -> np.ndarray:
    e_rmse = np.nan
    f_rmse = np.nan
    if state0 == NullState or state1 == NullState:
        pass
    elif state0.energy is None or state1.energy is None:
        pass
    else:
        e_rmse = np.abs(state0.per_atom_energy - state1.per_atom_energy)
        f_rmse = np.sqrt(
            np.mean((state0.per_atom.forces - state1.per_atom.forces) ** 2)
        )
    return np.array([e_rmse, f_rmse], dtype=float)


compute_error = python_app(_compute_error, executors=["default_threads"])


@typeguard.typechecked
def _exceeds_error(
    errors: np.ndarray,
    thresholds: np.ndarray,
) -> bool:
    return bool(np.any(errors > thresholds))


exceeds_error = python_app(_exceeds_error, executors=["default_threads"])


@typeguard.typechecked
def evaluate_outputs(
    outputs: list[SimulationOutput],
    hamiltonian: Hamiltonian,
    reference: Reference,
    identifier: Union[AppFuture, int],
    error_thresholds_for_reset: list[Optional[float]],
    error_thresholds_for_discard: list[Optional[float]],
    metrics: Metrics,
) -> tuple[Union[int, AppFuture], Dataset, list[AppFuture]]:
    states = [o.get_state() for o in outputs]  # take exit status into account
    eval_ref = [evaluate(s, reference) for s in states]
    eval_mod = Dataset(states).evaluate(hamiltonian)
    errors = [compute_error(s, eval_mod[i]) for i, s in enumerate(eval_ref)]
    processed_states = []
    resets = []
    for i, state in enumerate(eval_ref):
        error_discard = exceeds_error(
            errors[i],
            np.array(error_thresholds_for_discard, dtype=float),
        )
        error_reset = exceeds_error(
            errors[i],
            np.array(error_thresholds_for_reset, dtype=float),
        )
        reset = boolean_or(
            error_discard,
            error_reset,
            isnan(errors[i]),
        )

        _ = assign_identifier(state, identifier, reset)
        assigned = unpack_i(_, 0)
        identifier = unpack_i(_, 1)
        processed_states.append(assigned)
        resets.append(reset)

    metrics.log_walkers(outputs, errors, processed_states, resets)
    data = Dataset(processed_states).filter("identifier")
    return identifier, data, resets


@typeguard.typechecked
@psiflow.serializable
class Learning:
    reference: Reference
    path_output: str
    identifier: Union[AppFuture, int]
    train_valid_split: float
    error_thresholds_for_reset: list[Optional[float]]
    error_thresholds_for_discard: list[Optional[float]]
    metrics: Metrics
    iteration: int
    data: Dataset

    def __init__(
        self,
        reference: Reference,
        path_output: Union[str, Path],
        train_valid_split: float = 0.9,
        error_thresholds_for_reset: Union[list, tuple] = (0.02, 0.2),
        error_thresholds_for_discard: Union[list, tuple] = (0.03, 0.6),
        wandb_group: Optional[str] = None,
        wandb_project: Optional[str] = None,
        initial_data: Optional[Dataset] = None,
    ):
        self.reference = reference
        self.path_output = str(path_output)
        Path(self.path_output).mkdir(exist_ok=True, parents=True)
        self.train_valid_split = train_valid_split
        self.error_thresholds_for_reset = list(error_thresholds_for_reset)
        self.error_thresholds_for_discard = list(error_thresholds_for_discard)
        self.metrics = Metrics(
            wandb_group,
            wandb_project,
        )

        if initial_data is None:
            self.data = Dataset([])
            self.identifier = 0
        else:
            self.data = initial_data
            self.identifier = initial_data.assign_identifiers()
            self.metrics.update(self.data, Zero())

        self.iteration = -1

    def update(self, learning: Learning) -> None:
        assert self.path_output == learning.path_output
        self.train_valid_split = learning.train_valid_split
        self.error_thresholds_for_reset = learning.error_thresholds_for_reset
        self.error_thresholds_for_discard = learning.error_thresholds_for_discard
        self.metrics = learning.metrics
        self.reference = learning.reference
        self.data = learning.data
        self.identifier = learning.identifier
        self.iteration = learning.iteration

    def skip(self, name: str) -> bool:
        if not (Path(self.path_output) / name).exists():
            return False
        else:  # check that the iteration has completed, or delete
            if not (
                Path(self.path_output) / name / "_restart" / "learning.json"
            ).exists():
                shutil.rmtree(Path(self.path_output) / name)
                return False
            else:
                return True

    def save(self, name: str, model: Model, walkers: list[Walker]) -> None:
        model.save(Path(self.path_output) / name)
        self.data.save(Path(self.path_output) / name / "data.xyz")

        path = Path(self.path_output) / name / "_restart"
        path.mkdir(exist_ok=False, parents=True)
        for i, walker in enumerate(walkers):
            psiflow.serialize(
                walker,
                path_json=path / "{}.json".format(i),
                copy_to=path,
            )
        psiflow.serialize(model, path_json=path / "model.json", copy_to=path)
        psiflow.serialize(self, path_json=path / "learning.json", copy_to=path)
        psiflow.wait()

    def load(self, name: str) -> tuple[Model, list[Walker]]:
        path = Path(self.path_output) / name / "_restart"
        with open(path / "learning.json", "r") as f:
            learning_dict = f.read()
        self.update(psiflow.deserialize(learning_dict))
        with open(path / "model.json", "r") as f:
            model_dict = f.read()
        model = psiflow.deserialize(model_dict)

        i = 0
        walkers = []
        while True:
            path_walker = path / "{}.json".format(i)
            if path_walker.exists():
                with open(path_walker, "r") as f:
                    content = f.read()
                walkers.append(psiflow.deserialize(content))
                i += 1
            else:
                break
        return model, walkers

    def log(self, name):
        logger.log("\t+++  {}  +++".format(name))

    def passive_learning(
        self,
        model: Model,
        walkers: list[Walker],
        hamiltonian: Hamiltonian,
        steps: int,
        **sampling_kwargs,
    ) -> tuple[Model, list[Walker]]:
        self.iteration += 1
        name = "{}_{}".format(self.iteration, "passive_learning")
        if self.skip(name):
            model, walkers = self.load(name)
        else:
            backup = []
            for w in walkers:
                backup.append(w.hamiltonian)
                w.hamiltonian = w.hamiltonian + hamiltonian

            step = sampling_kwargs.pop("step", None)
            if step is None:
                step = steps
            nevaluations = steps // step
            for _i in range(nevaluations):
                outputs = sample(
                    walkers,
                    steps=step,
                    step=None,
                    **sampling_kwargs,
                )
                # only apply thresholds to forces, not to energies since large offset can exist
                threshold_reset = [None, self.error_thresholds_for_reset[1]]
                threshold_discard = [None, self.error_thresholds_for_discard[1]]
                identifier, data, _ = evaluate_outputs(  # ignore resets
                    outputs,
                    hamiltonian,
                    self.reference,
                    self.identifier,
                    threshold_reset,
                    threshold_discard,
                    self.metrics,
                )
                self.identifier = identifier
                self.data += data  # automatically sorted based on identifier

            train, valid = self.data.split(self.train_valid_split)  # also shuffles
            model.reset()
            model.initialize(train)
            model.train(train, valid)

            # log final errors of entire dataset
            self.metrics.update(self.data, model.create_hamiltonian())

            # restore original walker hamiltonians and apply conditional reset
            for w, h in zip(walkers, backup):
                w.hamiltonian = h

            self.save(name, model, walkers)
        return model, walkers

    def active_learning(
        self,
        model: Model,
        walkers: list[Walker],
        steps: int,
        **sampling_kwargs,
    ) -> tuple[Model, list[Walker]]:
        self.iteration += 1
        name = "{}_{}".format(self.iteration, "active_learning")
        if self.skip(name):
            model, walkers = self.load(name)
        else:
            hamiltonian = model.create_hamiltonian()
            backup = []
            for w in walkers:
                backup.append(w.hamiltonian)
                w.hamiltonian = w.hamiltonian + hamiltonian

            assert sampling_kwargs.get("step", None) is None
            outputs = sample(walkers, steps, **sampling_kwargs)
            identifier, data, resets = evaluate_outputs(
                outputs,
                hamiltonian,
                self.reference,
                self.identifier,
                self.error_thresholds_for_reset,
                self.error_thresholds_for_discard,
                self.metrics,
            )
            self.identifier = identifier
            self.data += data  # automatically sorted based on identifier

            train, valid = self.data.split(self.train_valid_split)  # also shuffles
            model.reset()
            model.initialize(train)
            model.train(train, valid)

            # log final errors of entire dataset
            self.metrics.update(self.data, model.create_hamiltonian())

            # restore original walker hamiltonians and apply conditional reset
            for i, (w, h) in enumerate(zip(walkers, backup)):
                w.hamiltonian = h
                w.reset(resets[i])

            self.save(name, model, walkers)
        return model, walkers
