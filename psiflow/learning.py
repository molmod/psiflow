from __future__ import annotations  # necessary for type-guarding class methods

import shutil
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture, Future

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry#, NullState, assign_identifier
from psiflow.hamiltonians import Hamiltonian, Zero
from psiflow.metrics import Metrics
from psiflow.models import MACE
# from psiflow.models import Model
from psiflow.reference import Reference
from psiflow.sampling import SimulationOutput, Walker, sample
from psiflow.utils.apps import boolean_or, isnan


logger = logging.getLogger(__name__)  # logging per module


# TODO: reset_model method to remove existing MLP + train/val split
# TODO: some plugin for data discarding + walker reset
# TODO: some plugin for data selection / uncertainty thing
# TODO: some plugin for wandb
# TODO: add_data method after init?
# TODO: clean_data method to throw away ridiculous structures
# TODO: remove psiflow.wait calls -- but still have blocking functionality

# TODO: PASSIVE LEARNING
#  1. walk MD + store structures + track which structure belongs to which walker
#  2. some data selection / cleanup step
#  3. reference single point + hamiltonian eval
#  4. compare geometries + filter + discard + keep track of walkers to reset
#  5. update train/val and retrain



# TODO: remove -> merge into one app
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

# TODO: remove -> merge into one app
@typeguard.typechecked
def _exceeds_error(
    errors: np.ndarray,
    thresholds: np.ndarray,
) -> bool:
    return bool(np.any(errors > thresholds))


exceeds_error = python_app(_exceeds_error, executors=["default_threads"])

# TODO: remove -> merge into one app
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
    eval_ref = [reference.evaluate(s) for s in states]
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


class Thresholds:
    """Container for hyperparams"""
    pass

@psiflow.register_serializable
class Learning:
    root: Path
    model: MACE
    reference: Reference
    train_valid_split: float
    identifier: Union[AppFuture, int]
    error_tresholds: Optional[Thresholds]
    # error_thresholds_for_reset: list[Optional[float]]
    # error_thresholds_for_discard: list[Optional[float]]
    # metrics: Metrics
    iteration: int
    wait_for: Future
    # data: Dataset

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
        # TODO: accept model argument
        # TODO: differentiate between fresh start <> restart
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
        # TODO: does this make sense?
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
        # TODO: upon restart, should we not immediately find how far we got and what the next action should be?
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
        # TODO: fully revamp this method
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
        # TODO: remove model argument + return only walkers
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
                # TODO: splits into multiple simulations and does data evaluation + validation in steps
                #  -- do not think we want this.. The instruction is to collect multiple geometries per walker
                #  -- because we should handle all data validation in one go (errors / uncertainty ...)
                outputs = sample(
                    walkers,
                    steps=step,
                    step=None,
                    **sampling_kwargs,
                )
                # only apply thresholds to forces, not to energies since large offset can exist
                # TODO: give users the choice?
                threshold_reset = [None, self.error_thresholds_for_reset[1]]
                threshold_discard = [None, self.error_thresholds_for_discard[1]]
                # TODO: do single point evaluations here
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

            # TODO: keep consistent train/val splits and do not reset model
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
