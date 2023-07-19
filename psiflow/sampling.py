from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, NamedTuple
import typeguard
import logging
from typing import Optional, Union
from collections import OrderedDict
import numpy as np

from parsl.dataflow.futures import AppFuture
from parsl.app.app import join_app, python_app

from psiflow.data import Dataset, FlowAtoms, NullState
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.walkers import BaseWalker
from psiflow.utils import unpack_i, copy_app_future
from psiflow.committee import Committee


logger = logging.getLogger(__name__) # logging per module


formatted_keys = {
        'counter'    : lambda x: '{:7} steps'.format(x),
        'temperature': lambda x: 'avg temp [K]: {:<6.1f}'.format(x),
        'time'       : lambda x: 'elapsed time [s]: {:<7.1f}'.format(x),
        'reset'      : lambda x: 'reset: {:<5}'.format(str(x)),
        }
            

@join_app
@typeguard.typechecked
def log_evaluation(
        i: int,
        metadata: NamedTuple,
        state: FlowAtoms,
        errors: Optional[tuple[float, float]],
        condition: bool,
        identifier: int,
        ) -> AppFuture:
    from pathlib import Path
    from psiflow.utils import copy_app_future
    s = 'WALKER {:>5}:'.format(i)
    metadata_dict = metadata._asdict()
    for key in metadata_dict:
        if (key in formatted_keys) and (key != 'reset'):
            s += formatted_keys[key](metadata_dict[key].result())
            s += ' ' * 8
        else:
            pass
    s += '\n'
    if metadata.reset.result():
        assert state == NullState
        assert metadata.state.result() == NullState
        s += '\tfailed during propagation; no evaluation was attempted'.format(i)
        s += '\n\twalker reset'
        assert condition
    else:
        if not state.reference_status:
            s += '\tevaluation failed; see {} in the task_logs directory'.format(Path(state.reference_stderr).stem)
            assert condition
        else:
            s += '\tevaluation OK; state received unique id {}'.format(identifier - 1)
            assert errors is not None
        s += '\n'
        if errors is not None:
            #assert not errors[0] == 0.0
            #assert not errors[1] == 0.0
            if condition:
                s += '\tenergy/force RMSE: {:7.1f} meV/atom  | {:5.0f} meV/A -- above threshold; walker reset'.format(*errors)
            else:
                s += '\tenergy/force RMSE: {:7.1f} meV/atom  | {:5.0f} meV/A -- below threshold'.format(*errors)
                #s += '\tRMSE below threshold   ({:7.1f} meV/atom | {:5.0f} meV/A); walker OK'.format(*errors)
        else:
            s += '\twalker reset'
    s += '\n'
    logger.info(s)
    return copy_app_future(s)


@typeguard.typechecked
def _assign_identifier(state: FlowAtoms, identifier: int):
    if not state == NullState:
        if state.reference_status:
            state.info['identifier'] = identifier
            identifier += 1
        else:
            pass
    return state, identifier
assign_identifier = python_app(_assign_identifier, executors=['default'])


@typeguard.typechecked
def _compute_error(
        atoms0: FlowAtoms,
        atoms1: FlowAtoms,
        ) -> Optional[tuple[float, float]]:
    import numpy as np
    from psiflow.data import read_dataset
    from psiflow.utils import compute_error
    if not (atoms0 == NullState) and not (atoms1 == NullState):
        if atoms1.reference_status:
            assert np.allclose(atoms0.positions, atoms1.positions)
            error = compute_error(
                    atoms0,
                    atoms1,
                    None,
                    None,
                    metric='rmse',
                    properties=['energy', 'forces'],
                    )
            return error
    return None
compute_error = python_app(_compute_error, executors=['default'])


@typeguard.typechecked
def _check_error(
        error: Union[tuple[float, float], np.ndarray, None],
        error_thresholds: Union[tuple[float, float], np.ndarray],
        ) -> bool:
    if (error is None) or (not np.all(np.array(error) < np.array(error_thresholds))):
        return True # do reset in this case
    return False
check_error = python_app(_check_error, executors=['default'])


@join_app
@typeguard.typechecked
def evaluate(state: FlowAtoms, reset: bool, reference: BaseReference) -> AppFuture:
    from copy import deepcopy # deepcopy necessary to avoid modifying argument itself
    if not reset:
        assert not state == NullState
        return reference.evaluate(deepcopy(state))
    else:
        return copy_app_future(NullState)


def sample(
        identifier: Union[AppFuture, int],
        model: BaseModel,
        reference: BaseReference,
        walkers: list[BaseWalker],
        error_thresholds_for_reset: tuple[float, float] = (10, 200), # (e_rmse, f_rmse)
        ) -> tuple[Dataset, AppFuture]:
    assert len(error_thresholds_for_reset) == 2
    logger.info('')
    logger.info('')
    metadatas = [w.propagate(model) for w in walkers]
    states  = [evaluate(m.state, m.reset, reference) for m in metadatas]
    labeled = model.evaluate(Dataset([m.state for m in metadatas])) # evaluate in batches
    errors  = [compute_error(labeled[i], state) for i, state in enumerate(states)]
    for i in range(len(states)):
        f = assign_identifier(states[i], identifier)
        state      = unpack_i(f, 0)
        identifier = unpack_i(f, 1)
        states[i]  = state
        condition  = check_error(
                error=errors[i],
                error_thresholds=error_thresholds_for_reset,
                )
        walkers[i].reset(condition)
        s = log_evaluation(i, metadatas[i], states[i], errors[i], condition, identifier)
        print(s.result())
    return Dataset(states).success(), identifier
