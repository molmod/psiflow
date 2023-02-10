from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
import logging
from pathlib import Path

from parsl.app.app import join_app
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms
from psiflow.sampling import BaseWalker, PlumedBias
from psiflow.reference import BaseReference
from psiflow.models import BaseModel
from psiflow.utils import copy_app_future


logger = logging.getLogger(__name__) # logging per module


@join_app
@typeguard.typechecked
def generate(
        name: str,
        walker: BaseWalker,
        model: BaseModel,
        reference: BaseReference,
        *args, # waits for these futures to complete before execution
        bias: Optional[PlumedBias] = None,
        checks: Optional[list] = None,
        ) -> AppFuture:
    float32 = Path(model.deploy_future['float32'].filepath).stem
    float64 = Path(model.deploy_future['float64'].filepath).stem
    logger.info('\tpropagating walker {} using models: {}(float32) and '
            '{}(float64)'.format(name, float32, float64))
    state = walker.propagate(
            safe_return=False,
            bias=bias,
            model=model,
            keep_trajectory=False,
            )
    if checks is not None:
        for check in checks:
            state = check(state, walker.tag_future)
    return evaluate(name, walker, model, reference, state, bias=bias, checks=checks)


@join_app
@typeguard.typechecked
def evaluate(
        name: str,
        walker: BaseWalker,
        model: BaseModel,
        reference: BaseReference,
        *args, # waits for these futures to complete before execution
        bias: Optional[PlumedBias] = None,
        checks: Optional[list] = None,
        ) -> AppFuture:
    assert len(args) == 1
    state = args[0]
    if state is not None:
        logger.info('\tstate from walker {} OK; proceeding with evaluation using {}'.format(name, reference.__class__))
        return gather(
                name,
                walker,
                model,
                reference,
                reference.evaluate(state),
                bias=bias,
                checks=checks,
                )
    else:
        logger.info('\tstate from walker {} not OK; retrying'.format(name, reference.__class__))
        walker.reset(conditional=False)
        walker.parameters.seed += 1
        logger.info('\tstate from walker {} not OK; retrying with seed {}'.format(name, walker.parameters.seed))
        return generate(name, walker, model, reference, bias=bias, checks=checks)


@join_app
@typeguard.typechecked
def gather(
        name: str,
        walker: BaseWalker,
        model: BaseModel,
        reference: BaseReference,
        *args, # waits for these futures to complete before execution
        bias: Optional[PlumedBias] = None,
        checks: Optional[list] = None,
        ) -> Union[AppFuture, FlowAtoms]:
    assert len(args) == 1
    state = args[0]
    if state.reference_status: # evaluation successful
        logger.info('\tstate from walker {} evaluated successfully'.format(name))
        return copy_app_future(state) # join_app must return future??
    else:
        walker.reset(conditional=False)
        walker.parameters.seed += 1
        logger.info('\tstate from walker {} failed during evaluation; retrying with seed {}'.format(name, walker.parameters.seed))
        return generate(name, walker, model, reference, bias=bias, checks=checks)


@typeguard.typechecked
class Generator:

    def __init__(
            self,
            name: str,
            walker: BaseWalker,
            reference: BaseReference,
            bias: Optional[PlumedBias] = None,
            ) -> None:
        self.name = name
        self.walker = walker
        self.reference = reference
        self.bias = bias

    def __call__(self, model, checks=None, wait_for_it=[]):
        return generate(
                self.name,
                self.walker,
                model,
                self.reference,
                *wait_for_it,
                bias=self.bias,
                checks=checks,
                )

    def multiply(self, ngenerators):
        generators = []
        for i in range(ngenerators):
            walker_ = self.walker.copy()
            walker_.parameters.seed = i
            if self.bias is not None:
                bias_ = self.bias.copy()
            else:
                bias_ = None
            # no need to copy reference
            generators.append(Generator(str(i), walker_, self.reference, bias_))
        return generators
