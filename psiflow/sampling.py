from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
import logging
from typing import Optional, Union

from parsl.dataflow.futures import AppFuture

from parsl.app.app import join_app, python_app
from psiflow.data import Dataset, FlowAtoms
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.walkers import BaseWalker
from psiflow.utils import unpack_i
from psiflow.committee import Committee


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


@join_app
def log_propagation(*counters: int) -> None:
    logger.info('PROPAGATION')
    for i, counter in enumerate(counters):
        s = '\twalker {:5}:'.format(i)
        if counter == 0:
            s += '\tfailed'
        else:
            s += '\t{:10} steps'.format(counter)
            

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
def assign_identifier(state: FlowAtoms, identifier: int):
    if state.reference_status:
        state.info['identifier'] = identifier
        identifier += 1
    else:
        pass
    return state, identifier


@join_app
def _sample(
        identifier: int,
        model: BaseModel,
        reference: BaseReference,
        walkers: list[BaseWalker],
        error_thresholds_for_reset: tuple[float] = (10, 200), # (e_rmse, f_rmse)
        ) -> list[AppFuture]:
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
    return [data.get(indices=data.success).as_list(), identifier]


# use wrapper because join_app can only return future
def sample(*args, **kwargs) -> tuple[Dataset, AppFuture]:
    f = _sample(*args, **kwargs)
    return Dataset(unpack_i(f, 0)), unpack_i(f, 1)
