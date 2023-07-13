from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
import logging
from typing import Optional, Union

from parsl.app.app import join_app, python_app
from psiflow.data import Dataset, FlowAtoms
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.walkers import BaseWalker
from psiflow.utils import unpack_i
from psiflow.committee import Committee


logger = logging.getLogger(__name__) # logging per module


@join_app
def log_propagation(*counters: int) -> None:
    logger.info('PROPAGATION')
    for i, counter in enumerate(counters):
        logger.info('\twalker {:5}:\t{:10} steps'.format(i, counter))


@join_app
def log_evaluation(*states: FlowAtoms) -> None:
    logger.info('QM EVALUATION')
    for i, state in enumerate(states):
        if state.reference_status:
            assert 'identifier' in state.info.keys()
            logger.info('\tstate from walker {:5}: \tsuccessful\t(received id {})'.format(i, state.info['identifier']))
        else:
            logger.info('\tstate from walker {:5}: \tfailed'.format(i))


@python_app(executors=['default'])
def assign_identifier(identifier: int, state: FlowAtoms):
    if state.reference_status:
        state.info['identifier'] = identifier
        identifier += 1
    else:
        pass
    return identifier, state


#@typeguard.typechecked
@join_app
def sample(
        identifier: int,
        model: BaseModel,
        reference: BaseReference,
        walkers: list[BaseWalker],
        error_thresholds_for_reset: tuple[float] = (10, 200), # (e_rmse, f_rmse)
        ) -> Dataset:
    states = [w.propagate(model) for w in walkers]
    log_propagation(*[w.counter_future for w in walkers])
    states = [reference.evaluate(s) for s in states]
    for i in range(len(states)):
        f = assign_identifier(states[i], identifier)
        state      = unpack_i(f, 0)
        identifier = unpack_i(f, 1)
        states[i] = state
    log_evaluation(*states)
    data = Dataset(states)
    return data.get(indices=data.success), identifier
