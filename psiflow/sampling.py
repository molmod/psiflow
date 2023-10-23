from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import NamedTuple, Optional, Union

import numpy as np
import typeguard
from parsl.app.app import join_app, python_app
from parsl.dataflow.futures import AppFuture

from psiflow.committee import Committee, filter_disagreements
from psiflow.data import Dataset, FlowAtoms, NullState
from psiflow.metrics import Metrics
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.utils import copy_app_future, unpack_i
from psiflow.walkers import BaseWalker

logger = logging.getLogger(__name__)  # logging per module


formatted_keys = {
    "counter": lambda x: "{:7} steps".format(x),
    "temperature": lambda x: "avg temp [K]: {:<6.1f}".format(x),
    "time": lambda x: "elapsed time [s]: {:<7.1f}".format(x),
    "reset": lambda x: "reset: {:<5}".format(str(x)),
}


@typeguard.typechecked
def log_metadata(
    i: int,
    metadata: NamedTuple,
    state: FlowAtoms,
    condition: bool,
) -> str:
    from pathlib import Path

    s = "WALKER {:>5}:".format(i)
    metadata_dict = metadata._asdict()
    for key in metadata_dict:
        if (key in formatted_keys) and (key != "reset"):
            s += formatted_keys[key](metadata_dict[key].result())
            s += " " * 8
        else:
            pass
    s += "\n"
    if metadata.reset.result():
        assert state == NullState
        assert metadata.state.result() == NullState
        s += "\tpropagation failed"
        if "stdout" in metadata_dict:
            s += "; see {} in the task_logs directory".format(
                Path(metadata.stdout).stem
            )
        s += "\n\twalker reset\n"
        assert condition
    return s


@typeguard.typechecked
def _assign_identifier(state: FlowAtoms, identifier: int):
    if not state == NullState:
        if state.reference_status:
            state.info["identifier"] = identifier
            identifier += 1
        else:
            pass
    return state, identifier


assign_identifier = python_app(_assign_identifier, executors=["Default"])


@join_app
@typeguard.typechecked
def log_evaluation_model(
    i: int,
    metadata: NamedTuple,
    state: FlowAtoms,
    errors: tuple[Optional[float], Optional[float]],
    condition: bool,
    identifier: int,
) -> AppFuture:
    from pathlib import Path

    from psiflow.utils import copy_app_future

    s = log_metadata(i, metadata, state, condition)
    if not metadata.reset.result():
        if not state.reference_status:
            s += "\tevaluation failed; see {} in the task_logs directory".format(
                Path(state.reference_stderr).stem
            )
            assert condition
        else:
            s += "\tevaluation successful; state received unique id {}".format(
                identifier - 1
            )
            assert errors is not None
        s += "\n"
        if errors[0] is not None:
            if condition:
                s += "\tenergy/force RMSE: {:7.1f} meV/atom  | {:5.0f} meV/A (above threshold)\n\twalker reset".format(
                    *errors
                )
            else:
                s += "\tenergy/force RMSE: {:7.1f} meV/atom  | {:5.0f} meV/A (below threshold)".format(
                    *errors
                )
        else:
            s += "\twalker reset"
        s += "\n"
    logger.info(s)
    return copy_app_future(s)


@typeguard.typechecked
def _compute_error(
    atoms0: FlowAtoms,
    atoms1: FlowAtoms,
) -> tuple[Optional[float], Optional[float]]:
    import numpy as np

    from psiflow.utils import compute_error

    if not (atoms0 == NullState) and not (atoms1 == NullState):
        if atoms1.reference_status:
            assert np.allclose(atoms0.positions, atoms1.positions)
            error = compute_error(
                atoms0,
                atoms1,
                metric="rmse",
                mask=np.array([True] * len(atoms0)),
                properties=["energy", "forces"],
            )
            return error
    return None, None


compute_error = python_app(_compute_error, executors=["Default"])


@typeguard.typechecked
def _check_error(
    error: tuple[Optional[float], Optional[float]],
    error_thresholds: tuple[float, float],
) -> bool:
    if (
        (error[0] is None)
        or (error[1] is None)
        or (not np.all(np.array(error) < np.array(error_thresholds)))
    ):
        return True  # do reset in this case
    return False


check_error = python_app(_check_error, executors=["Default"])


@typeguard.typechecked
def sample_with_model(
    model: BaseModel,
    reference: BaseReference,
    walkers: list[BaseWalker],
    identifier: Union[AppFuture, int],
    error_thresholds_for_reset: tuple[float, float] = (10, 200),  # (e_rmse, f_rmse)
    metrics: Optional[Metrics] = None,
) -> tuple[Dataset, AppFuture]:
    assert len(error_thresholds_for_reset) == 2
    logger.info("")
    logger.info("")
    metadatas = [w.propagate(model) for w in walkers]
    states = [reference.evaluate(m.state) for m in metadatas]
    labeled = model.evaluate(
        Dataset([m.state for m in metadatas])
    )  # evaluate in batches
    errors = [compute_error(labeled[i], state) for i, state in enumerate(states)]
    for i in range(len(states)):
        f = assign_identifier(states[i], identifier)
        state = unpack_i(f, 0)
        identifier = unpack_i(f, 1)
        states[i] = state
        condition = check_error(
            error=errors[i],
            error_thresholds=error_thresholds_for_reset,
        )
        walkers[i].reset(condition)
        log_evaluation_model(
            i, metadatas[i], states[i], errors[i], condition, identifier
        )
        if metrics is not None:
            metrics.log_walker(
                i, walkers[i], metadatas[i], states[i], errors[i], condition, identifier
            )
    return Dataset(states).labeled(), identifier


@join_app
@typeguard.typechecked
def evaluate_subset(
    state: FlowAtoms, reference: BaseReference, i: int, indices: np.ndarray
) -> AppFuture[FlowAtoms]:
    if i in indices:
        return reference.evaluate(state)
    else:
        return copy_app_future(NullState)


@typeguard.typechecked
def _reset_condition(
    i: int,
    state: FlowAtoms,
    indices: np.ndarray,
    checked_error: bool,
) -> bool:
    from psiflow.data import NullState

    if (i in indices) or (state == NullState) or checked_error:
        return True
    return False


reset_condition = python_app(_reset_condition, executors=["Default"])


@join_app
@typeguard.typechecked
def log_evaluation_committee(
    i: int,
    metadata: NamedTuple,
    state: FlowAtoms,
    errors: tuple[Optional[float], Optional[float]],
    condition: bool,
    identifier: int,
    disagreement: float,
    indices: np.ndarray,
) -> AppFuture:
    from pathlib import Path

    from psiflow.utils import copy_app_future

    s = log_metadata(i, metadata, state, condition)
    if not metadata.reset.result():
        s += "\tcommittee disagreement: RMSE(F - mean(F)) = {} eV/A".format(
            disagreement
        )
        if i in indices:
            s += " (high)\n"  # state selected for evaluation
            assert condition
            if not state.reference_status:
                s += "\tevaluation failed; see {} in the task_logs directory".format(
                    Path(state.reference_stderr).stem
                )
            else:
                assert errors[0] is not None
                assert errors[1] is not None
                s += "\tevaluation successful; state received unique id {}\n".format(
                    identifier - 1
                )
                s += "\tenergy/force RMSE: {:7.1f} meV/atom  | {:5.0f} meV/A\n".format(
                    *errors
                )
                if condition:
                    s += "\twalker reset"
        else:
            s += " (low)\n\tevaluation skipped"  # state not selected
            assert errors[0] is None
        s += "\n"
    logger.info(s)
    return copy_app_future(s)


@typeguard.typechecked
def sample_with_committee(
    committee: Committee,
    reference: BaseReference,
    walkers: list[BaseWalker],
    identifier: Union[AppFuture, int],
    nstates: int,
    error_thresholds_for_reset: tuple[float, float] = (10, 200),  # (e_rmse, f_rmse)
    metrics: Optional[Metrics] = None,
) -> tuple[Dataset, AppFuture]:
    logger.info("")
    logger.info("")
    metadatas = [w.propagate(committee.models[0]) for w in walkers]
    states = [m.state for m in metadatas]
    disagreements = committee.compute_disagreements(Dataset(states))
    indices = filter_disagreements(disagreements, nstates)
    states = [evaluate_subset(s, reference, i, indices) for i, s in enumerate(states)]

    # compute errors for state which were evaluated
    labeled = committee.models[0].evaluate(Dataset([m.state for m in metadatas]))
    errors = [compute_error(labeled[i], state) for i, state in enumerate(states)]
    for i in range(len(metadatas)):
        f = assign_identifier(states[i], identifier)
        state = unpack_i(f, 0)
        identifier = unpack_i(f, 1)
        states[i] = state
        checked_error = check_error(
            error=errors[i],
            error_thresholds=error_thresholds_for_reset,
        )
        condition = reset_condition(i, states[i], indices, checked_error)
        walkers[i].reset(condition)
        log_evaluation_committee(
            i,
            metadatas[i],
            states[i],
            errors[i],
            condition,
            identifier,
            unpack_i(disagreements, i),
            indices,
        )
        if metrics is not None:
            metrics.log_walker(
                i,
                walkers[i],
                metadatas[i],
                states[i],
                errors[i],
                condition,
                identifier,
                unpack_i(disagreements, i),
            )
    return Dataset(states).labeled(), identifier
