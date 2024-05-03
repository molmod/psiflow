from __future__ import annotations  # necessary for type-guarding class methods

import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, assign_identifier
from psiflow.geometry import Geometry, NullState
from psiflow.hamiltonians import Hamiltonian
from psiflow.metrics import Metrics
from psiflow.models import Model
from psiflow.reference import Reference
from psiflow.sampling import SimulationOutput, Walker, sample
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
        e_rmse = np.abs(state0.per_atom_energy - state1.per_atom_energy)
        f_rmse = np.sqrt(
            np.mean((state0.per_atom.forces - state1.per_atom.forces) ** 2)
        )
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
    hamiltonian: Hamiltonian,
    reference: Reference,
    identifier: Union[AppFuture, int],
    error_thresholds_for_reset: tuple[float, float],
    error_thresholds_for_discard: tuple[float, float],
    metrics: Metrics,
) -> Dataset:
    states = [o.get_state() for o in outputs]  # take exit status into account
    eval_ref = [reference.evaluate(s) for s in states]
    eval_mod = hamiltonian.evaluate(Dataset(states))
    errors = [compute_error(s, eval_mod[i]) for i, s in enumerate(eval_ref)]
    processed_states = []
    resets = []
    for i, state in enumerate(eval_ref):
        discard = exceeds_error(errors[i], error_thresholds_for_discard)
        reset = exceeds_error(errors[i], error_thresholds_for_reset)
        resets.append(reset)

        _ = assign_identifier(state, identifier, discard)
        assigned = unpack_i(_, 0)
        identifier = unpack_i(_, 1)
        processed_states.append(assigned)

    metrics.log_walkers(
        outputs,
        errors,
        eval_ref,
        resets,
    )
    data = Dataset(processed_states).filter("identifier")
    return identifier, data, resets


@typeguard.typechecked
@psiflow.serializable
class Learning:
    reference: Reference
    path_output: Path
    identifier: Union[AppFuture, int]
    train_valid_split: float
    error_thresholds_for_reset: tuple[float, float]
    error_thresholds_for_discard: tuple[float, float]
    metrics: Metrics
    iteration: int

    def __init__(
        self,
        reference: Reference,
        path_output: Union[str, Path],
        train_valid_split: float = 0.9,
        error_thresholds_for_reset: tuple[float, float] = (20, 300),
        error_thresholds_for_discard: tuple[float, float] = (30, 600),
        wandb_project: Optional[str] = None,
        wandb_group: Optional[str] = None,
        initial_data: Optional[Dataset] = None,
    ):
        self.reference = reference
        self.path_output = Path(path_output)
        self.path_output.mkdir(exist_ok=False, parents=True)
        self.train_valid_split = train_valid_split
        self.error_thresholds_for_reset = error_thresholds_for_reset
        self.error_thresholds_for_discard = error_thresholds_for_discard
        self.metrics = Metrics(
            wandb_project,
            wandb_group,
        )

        if initial_data is None:
            self.data = Dataset([])
            self.identifier = 0
        else:
            self.data = initial_data
            self.identifier = initial_data.assign_identifiers()

        self.iteration = -1

    def update(self, learning: Learning) -> None:
        assert str(self.path_output) == str(learning.path_output)
        self.train_valid_split = learning.train_valid_split
        self.error_thresholds_for_reset = learning.error_thresholds_for_reset
        self.error_thresholds_for_discard = learning.error_thresholds_for_discard
        self.metrics = learning.metrics
        self.reference = learning.reference
        self.data = learning.data
        self.identifier = learning.identifier
        self.iteration = learning.iteration

    def skip(self, name: str) -> bool:
        if not (self.path_output / name).exists():
            return False
        else:  # check that the iteration has completed, or delete
            if not (self.path_output / name / "_restart" / "learning.json").exists():
                shutil.rmtree(self.path_output / name)
                return False
            else:
                return True

    def save(self, name: str, model: Model, walkers: list[Walker]) -> None:
        model.save(self.path_output / name)
        self.data.save(self.path_output / name / "data.xyz")

        path = self.path_output / name / "_restart"
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

    def load(self, name: str) -> None:
        path = self.path_output / name / "_restart"
        self.update(psiflow.deserialize(path / "learning.json"))
        model = psiflow.deserialize(path / "model.json")

        i = 0
        walkers = []
        while True:
            path_walker = path / "{}.json".format(i)
            if path_walker.exists():
                walkers.append(psiflow.deserialize(path_walker))
        return model, walkers

    def pretraining(
        self,
        walkers: list[Walker],
        hamiltonian: Hamiltonian,
        steps: int,
        **sampling_kwargs,
    ) -> tuple[Model, list[Walker]]:
        self.iteration += 1
        name = "{}_{}".format(self.iteration, "pretraining")
        if self.skip(name):
            model, walkers = self.load(name)
        else:
            backup = []
            for w in walkers:
                backup.append(w.hamiltonian)
                w.hamiltonian = w.hamiltonian + hamiltonian

            step = sampling_kwargs.get("step", None)
            if step is None:
                step = 1
            nevaluations = steps // step
            for _i in range(nevaluations):
                outputs = sample(walkers, steps, **sampling_kwargs)
                identifier, data, _ = evaluate_outputs(  # ignore resets
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

            # restore original walker hamiltonians and apply conditional reset
            for w, h in zip(walkers, backup):
                w.hamiltonian = h

            self.save(name, model, walkers)
        return model, walkers

    def sample_qm_train(
        self,
        walkers: list[Walker],
        steps: int,
        **sampling_kwargs,
    ) -> tuple[Model, list[Walker]]:
        self.iteration += 1
        name = "{}_{}".format(self.iteration, "sample_qm_train")
        if self.skip(name):
            model, walkers = self.load(name)
        else:
            hamiltonian = model.create_hamiltonian()
            backup = []
            for w in walkers:
                backup.append(w.hamiltonian)
                w.hamiltonian = w.hamiltonian + hamiltonian

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

            # restore original walker hamiltonians and apply conditional reset
            for i, (w, h) in enumerate(zip(walkers, backup)):
                w.hamiltonian = h
                w.reset(resets[i])

            self.save(name, model, walkers)
        return model, walkers
