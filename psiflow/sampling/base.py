from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Tuple
import typeguard
from dataclasses import dataclass, asdict
from copy import deepcopy
from pathlib import Path

from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.execution import ModelExecutionDefinition, Container, \
        ExecutionContext
from psiflow.utils import copy_app_future, unpack_i, copy_data_future, \
        save_yaml
from psiflow.data import save_atoms, FlowAtoms, Dataset


@typeguard.typechecked
def _app_safe_return(
        state: FlowAtoms,
        start: FlowAtoms,
        tag: str,
        ) -> FlowAtoms:
    if tag == 'unsafe':
        return start
    else:
        return state
app_safe_return = python_app(_app_safe_return, executors=['default'])


@typeguard.typechecked
def _is_reset(state: FlowAtoms, start: FlowAtoms) -> bool:
    if state == start: # positions, numbers, cell, pbc
        return True
    else:
        return False
is_reset = python_app(_is_reset, executors=['default'])


@typeguard.typechecked
def _update_tag(existing_tag: str, new_tag: str) -> str:
    if (existing_tag == 'safe') and (new_tag == 'safe'):
        return 'safe'
    else:
        return 'unsafe'
update_tag = python_app(_update_tag, executors=['default'])


@typeguard.typechecked
@dataclass
class EmptyParameters:
    pass


@typeguard.typechecked
class BaseWalker(Container):
    parameters_cls = EmptyParameters

    def __init__(
            self,
            context: ExecutionContext,
            atoms: Union[FlowAtoms, AppFuture],
            **kwargs,
            ) -> None:
        super().__init__(context)
        self.context = context

        # futures
        self.start_future = copy_app_future(atoms) # necessary!
        self.state_future = copy_app_future(atoms)
        self.tag_future   = copy_app_future('safe')

        # parameters
        self.parameters = self.parameters_cls(**deepcopy(kwargs))

    def propagate(
            self,
            safe_return: bool = False,
            keep_trajectory: bool = False,
            **kwargs,
            ) -> Union[AppFuture, Tuple[AppFuture, Dataset]]:
        app = self.context.apps(self.__class__, 'propagate')
        if keep_trajectory:
            file = self.context.new_file('data_', '.xyz')
        else:
            file = None
        result = app(
                self.state_future,
                deepcopy(self.parameters),
                keep_trajectory=keep_trajectory,
                file=file,
                **kwargs, # Model or Bias instance
                )
        self.state_future = unpack_i(result, 0)
        self.tag_future   = update_tag(self.tag_future, unpack_i(result, 1))
        if safe_return: # only return state if safe, else return start
            # this does NOT reset the walker!
            future = app_safe_return(
                    self.state_future,
                    self.start_future,
                    self.tag_future,
                    )
        else:
            future = self.state_future
        future = copy_app_future(future) # necessary
        if keep_trajectory:
            return future, Dataset(self.context, None, data_future=result.outputs[0])
        else:
            return future

    def tag_unsafe(self) -> None:
        self.tag_future = copy_app_future('unsafe')

    def tag_safe(self) -> None:
        self.tag_future = copy_app_future('safe')

    def reset_if_unsafe(self) -> None:
        self.state_future = app_safe_return(
                self.state_future,
                self.start_future,
                self.tag_future,
                )
        self.tag_future = copy_app_future('safe')

    def is_reset(self) -> AppFuture:
        return is_reset(self.state_future, self.start_future)

    def reset(self) -> AppFuture:
        self.state_future = copy_app_future(self.start_future)
        self.tag = copy_app_future('safe')
        return self.state_future

    def copy(self) -> BaseWalker:
        walker = self.__class__(
                self.context,
                self.state_future,
                )
        walker.start_future = copy_app_future(self.start_future)
        walker.tag_future   = copy_app_future(self.tag_future)
        walker.parameters   = deepcopy(self.parameters)
        return walker

    def save(
            self,
            path: Union[Path, str],
            require_done: bool = True,
            ) -> Tuple[DataFuture, DataFuture, DataFuture]:
        path = Path(path)
        assert path.is_dir()
        name = self.__class__.__name__
        path_start = path / 'start.xyz'
        path_state = path / 'state.xyz'
        path_pars  = path / (name + '.yaml')
        future_start = save_atoms(
                self.start_future,
                outputs=[File(str(path_start))],
                ).outputs[0]
        future_state = save_atoms(
                self.state_future,
                outputs=[File(str(path_state))],
                ).outputs[0]
        future_pars = save_yaml(
                asdict(self.parameters),
                outputs=[File(str(path_pars))],
                ).outputs[0]
        if require_done:
            future_start.result()
            future_state.result()
            future_pars.result()
        return future_start, future_state, future_pars

    @classmethod
    def create_apps(cls, context: ExecutionContext) -> None:
        assert not (cls == BaseWalker) # should never be called directly
        pass
