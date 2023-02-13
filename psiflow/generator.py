from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
import logging
from pathlib import Path
import os

from parsl.app.app import join_app
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms, Dataset
from psiflow.sampling import BaseWalker, PlumedBias, load_walker
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
        *args, # waits for these futures to complete before execution
        bias: Optional[PlumedBias] = None,
        checks: Optional[list] = None,
        ) -> AppFuture:
    if model is not None:
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
    return evaluate(name, walker, model, reference, state, walker.counter_future, bias=bias, checks=checks)


@join_app
@typeguard.typechecked
def evaluate(
        name: str,
        walker: BaseWalker,
        model: Optional[BaseModel],
        reference: Optional[BaseReference],
        *args, # waits for these futures to complete before execution
        bias: Optional[PlumedBias] = None,
        checks: Optional[list] = None,
        ) -> AppFuture:
    assert len(args) == 2
    state = args[0]
    counter = args[1]
    logger.info('\twalker {} propagated for {} steps'.format(name, counter))
    if state is not None:
        logger.info('\tstate from walker {} OK!'.format(name))
        if reference is not None:
            logger.info('\tevaluating state using {}'.format(name, reference.__class__))
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
            logger.info('\tno reference level given, returning state')
            return copy_app_future(state)
    else:
        logger.info('\tstate from walker {} not OK'.format(name, reference.__class__))
        logger.info('\tretrying with reset and different initial seed')
        walker.reset(conditional=False)
        walker.parameters.seed += 1
        return generate(name, walker, model, reference, bias=bias, checks=checks)


@join_app
@typeguard.typechecked
def gather(
        name: str,
        walker: BaseWalker,
        model: Optional[BaseModel],
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
            bias: Optional[PlumedBias] = None,
            ) -> None:
        self.name = name
        self.walker = walker
        self.bias = bias

    def __call__(
            self,
            model: Optional[BaseModel], # can be None when using RandomWalker
            reference: Optional[BaseReference],
            checks: Optional[list[Check]] = None,
            wait_for_it: list[AppFuture] = [],
            ) -> AppFuture:
        return generate(
                self.name,
                self.walker,
                model,
                reference,
                *wait_for_it,
                bias=self.bias,
                checks=checks,
                )

    def multiply(
            self,
            ngenerators: int,
            initialize_using: Optional[Dataset] = None,
            ) -> list[Generator]:
        generators = []
        if initialize_using is not None:
            length = initialize_using.length().result()
        for i in range(ngenerators):
            walker_ = self.walker.copy()
            walker_.parameters.seed = i
            if initialize_using is not None:
                walker_.state_future = copy_app_future(initialize_using[i % length])
                walker_.start_future = copy_app_future(initialize_using[i % length])
            if self.bias is not None:
                bias_ = self.bias.copy()
            else:
                bias_ = None
            generators.append(Generator(self.name + str(i), walker_, bias_))
        return generators


def load_generators(
        context: ExecutionContext,
        path: Union[Path, str],
        ) -> list[Generator]:
    path = Path(path)
    assert path.is_dir()
    generators = []
    for name in os.listdir(path):
        if not (path / name).is_dir():
            continue
        path_walker = path / name
        walker = load_walker(context, path_walker)
        path_plumed = path_walker / 'plumed_input.txt'
        if path_plumed.is_file(): # check if bias present
            bias = PlumedBias.load(context, path_walker)
        else:
            bias = None
        generators.append(Generator(name, walker, bias))
    return generators


def save_generators(
        generators: list[Generator],
        path: Union[Path, str],
        require_done: bool = True,
        ):
    path = Path(path)
    assert path.is_dir()
    for generator in generators:
        path_walker = path / generator.name
        path_walker.mkdir(parents=False)
        generator.walker.save(path_walker, require_done=require_done)
        if generator.bias is not None:
            generator.bias.save(path_walker, require_done=require_done)
