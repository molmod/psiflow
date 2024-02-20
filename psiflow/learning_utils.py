from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import Optional, Union

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


@typeguard.typechecked
def _assign_identifier(state: FlowAtoms, discard: bool, identifier: int):
    if discard:
        state = NullState
    if not state == NullState:
        if state.reference_status:
            state.info["identifier"] = identifier
            identifier += 1
    return state, identifier


assign_identifier = python_app(_assign_identifier, executors=["default_threads"])


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


compute_error = python_app(_compute_error, executors=["default_threads"])


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


check_error = python_app(_check_error, executors=["default_threads"])


@typeguard.typechecked
def sample_with_model(
    model: BaseModel,
    reference: BaseReference,
    walkers: list[BaseWalker],
    identifier: Union[AppFuture, int],
    error_thresholds_for_reset: tuple[float, float] = (10, 200),  # (e_rmse, f_rmse)
    error_thresholds_for_discard: tuple[float, float] = (20, 500),  # (e_rmse, f_rmse)
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
        discard = check_error(
            error=errors[i],
            error_thresholds=error_thresholds_for_discard,
        )
        condition = check_error(
            error=errors[i],
            error_thresholds=error_thresholds_for_reset,
        )
        f = assign_identifier(states[i], discard, identifier)
        state = unpack_i(f, 0)
        identifier = unpack_i(f, 1)
        states[i] = state
        if metrics is not None:
            metrics.log_walker(
                i,
                walkers[i],
                metadatas[i],
                states[i],
                errors[i],
                discard,
                condition,
                identifier,
            )
        walkers[i].reset(condition)
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


reset_condition = python_app(_reset_condition, executors=["default_threads"])


@typeguard.typechecked
def sample_with_committee(
    committee: Committee,
    reference: BaseReference,
    walkers: list[BaseWalker],
    identifier: Union[AppFuture, int],
    nstates: int,
    error_thresholds_for_reset: tuple[float, float] = (10, 200),  # (e_rmse, f_rmse)
    error_thresholds_for_discard: tuple[float, float] = (20, 500),  # (e_rmse, f_rmse)
    metrics: Optional[Metrics] = None,
) -> tuple[Dataset, AppFuture]:
    logger.info("")
    logger.info("")
    metadatas = [w.propagate(committee.models[0]) for w in walkers]
    states = [m.state for m in metadatas]
    disagreements = committee.compute_disagreements(Dataset(states))
    indices = filter_disagreements(disagreements, nstates)
    states = [evaluate_subset(s, reference, i, indices) for i, s in enumerate(states)]

    # compute errors for states which were evaluated
    labeled = committee.models[0].evaluate(Dataset([m.state for m in metadatas]))
    errors = [compute_error(labeled[i], state) for i, state in enumerate(states)]
    for i in range(len(metadatas)):
        discard = check_error(
            error=errors[i],
            error_thresholds=error_thresholds_for_discard,
        )
        checked_error = check_error(
            error=errors[i],
            error_thresholds=error_thresholds_for_reset,
        )
        f = assign_identifier(states[i], discard, identifier)
        state = unpack_i(f, 0)
        identifier = unpack_i(f, 1)
        states[i] = state
        condition = reset_condition(i, states[i], indices, checked_error)
        walkers[i].reset(condition)
        if metrics is not None:
            metrics.log_walker(
                i,
                walkers[i],
                metadatas[i],
                states[i],
                errors[i],
                discard,
                condition,
                identifier,
                unpack_i(disagreements, i),
            )
    return Dataset(states).labeled(), identifier
