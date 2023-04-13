from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
import logging
import numpy as np
from pathlib import Path
import os

from parsl.app.app import join_app
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms, Dataset, app_reset_atoms
from psiflow.sampling import BaseWalker, PlumedBias, load_walker, RandomWalker
from psiflow.reference import BaseReference
from psiflow.models import BaseModel
from psiflow.utils import copy_app_future
from psiflow.checks import Check


logger = logging.getLogger(__name__) # logging per module


@join_app
@typeguard.typechecked
def generate(
        name: str,
        walker: BaseWalker,
        model: Optional[BaseModel],
        reference: Optional[BaseReference],
        num_tries_sampling: int,
        num_tries_reference: int,
        *args, # waits for these futures to complete before execution
        checks: Optional[list[Check]] = None,
        ) -> AppFuture:
    for arg in args:
        assert not isinstance(arg, Check) # can occur by mistake
        assert not isinstance(arg, list) # can occur by mistake
    if not num_tries_sampling > 0:
        logger.info('reached max sampling retries for walker {}, aborting'.format(
            num_tries_sampling,
            name,
            ))
        return app_reset_atoms(walker.start_future)
    if not num_tries_reference > 0:
        logger.info('reached max reference retries for walker {}, aborting'.format(
            num_tries_reference,
            name,
            ))
        return app_reset_atoms(walker.start_future)
    logger.info('propagating walker {}\t ntries_sampling: {}\tntries_reference'
            ': {}'.format(name, num_tries_sampling, num_tries_reference))
    if model is not None: # can be none for random walker
        if 'float32' not in model.deploy_future.keys():
            print(model.deploy_future)
            raise ValueError('model most be deployed before calling generate')
        float32 = Path(model.deploy_future['float32'].filepath).stem
        float64 = Path(model.deploy_future['float64'].filepath).stem
        logger.info('\tusing models: {} (float32) and {} (float64)'.format(
                float32, float64))
    else:
        assert type(walker) == RandomWalker
    state = walker.propagate( # will reset if unsafe!
            model=model,
            keep_trajectory=False,
            )
    if checks is not None:
        for check in checks:
            state = check(state, walker.tag_future)
    return _evaluate(
            name,
            walker,
            model,
            reference,
            num_tries_sampling - 1, # performed one sampling try
            num_tries_reference,
            state,
            walker.counter_future,
            checks=checks,
            )


@join_app
@typeguard.typechecked
def _evaluate(
        name: str,
        walker: BaseWalker,
        model: Optional[BaseModel],
        reference: Optional[BaseReference],
        num_tries_sampling: int,
        num_tries_reference: int,
        *args, # waits for these futures to complete before execution
        checks: Optional[list] = None,
        ) -> AppFuture:
    assert len(args) == 2
    state = args[0]
    counter = args[1]
    logger.info('\twalker {} has a (total) counter value of {} steps'.format(name, counter))
    if state is not None:
        state.reset()
        logger.info('\tstate from walker {} OK!'.format(name))
        if reference is not None:
            logger.info('\tevaluating state using {}'.format(reference.__class__.__name__))
            return _gather(
                    name,
                    walker,
                    model,
                    reference,
                    num_tries_sampling,
                    num_tries_reference - 1, # one reference try
                    reference.evaluate(state),
                    checks=checks,
                    )
        else:
            logger.info('\tno reference level given, returning state')
            return copy_app_future(state)
    else:
        logger.info('\tstate from walker {} not OK'.format(name, reference.__class__))
        logger.info('\tresetting state and increment seed')
        walker.reset(conditional=False)
        walker.seed += 1
        return generate(
                name,
                walker,
                model,
                reference,
                num_tries_sampling,
                num_tries_reference,
                checks=checks,
                )


@join_app
@typeguard.typechecked
def _gather(
        name: str,
        walker: BaseWalker,
        model: Optional[BaseModel],
        reference: BaseReference,
        num_tries_sampling: int,
        num_tries_reference: int,
        *args, # waits for these futures to complete before execution
        checks: Optional[list] = None,
        ) -> Union[AppFuture, FlowAtoms]:
    assert len(args) == 1
    state = args[0]
    if state.reference_status: # evaluation successful
        logger.info('\tstate from walker {} evaluated successfully'.format(name))
        return copy_app_future(state) # join_app must return future??
    else:
        walker.reset(conditional=False)
        walker.seed += 1
        logger.info('\tstate from walker {} failed during evaluation; '
                'retrying with seed {}'.format(
                    name,
                    walker.seed,
                    ))
        return generate(
                name,
                walker,
                model,
                reference,
                num_tries_sampling,
                num_tries_reference,
                checks=checks,
                )


def generate_all(
        walkers: list[BaseWalker],
        model: Optional[BaseModel],
        reference: Optional[BaseReference],
        num_tries_sampling: int = 10,
        num_tries_reference: int = 1,
        checks: Optional[list[Check]] = None,
        ) -> Dataset:
    states = []
    for i, walker in enumerate(walkers):
        state = generate(
                str(i),
                walker,
                model,
                reference,
                num_tries_sampling,
                num_tries_reference,
                checks=checks, # call as keyword arg!
                )
        states.append(state)
    return Dataset(states)
