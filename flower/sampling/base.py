from dataclasses import dataclass, asdict
from copy import deepcopy
from pathlib import Path

from parsl.app.app import python_app
from parsl.data_provider.files import File

from flower.execution import ModelExecutionDefinition, Container
from flower.utils import copy_app_future, unpack_i, copy_data_future, \
        save_yaml, save_atoms


@python_app(executors=['default'])
def app_safe_return(state, start, tag):
    if tag == 'unsafe':
        return start
    else:
        return state


@python_app(executors=['default'])
def is_reset(state, start):
    if state == start: # positions, numbers, cell, pbc
        return True
    else:
        return False


@python_app
def update_tag(existing_tag, new_tag):
    if (existing_tag == 'safe') and (new_tag == 'safe'):
        return 'safe'
    else:
        return 'unsafe'


@dataclass
class EmptyParameters:
    pass


class BaseWalker(Container):
    parameters_cls = EmptyParameters

    def __init__(self, context, atoms, **kwargs):
        super().__init__(context)
        self.context = context

        # futures
        self.start_future = copy_app_future(atoms) # necessary!
        self.state_future = copy_app_future(atoms)
        self.tag_future   = copy_app_future('safe')

        # parameters
        self.parameters = self.parameters_cls(**deepcopy(kwargs))

    def propagate(self, safe_return=False, keep_trajectory=False, **kwargs):
        app = self.context.apps(self.__class__, 'propagate')
        result, dataset = app(
                self.state_future,
                deepcopy(self.parameters),
                **kwargs, # Model or Bias instance
                keep_trajectory=keep_trajectory,
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
            assert dataset is not None
            return future, dataset
        else:
            return future

    def tag_unsafe(self):
        self.tag_future = copy_app_future('unsafe')

    def tag_safe(self):
        self.tag_future = copy_app_future('safe')

    def reset_if_unsafe(self):
        self.state_future = app_safe_return(
                self.state_future,
                self.start_future,
                self.tag_future,
                )
        self.tag_future = copy_app_future('safe')

    def is_reset(self):
        return is_reset(self.state_future, self.start_future)

    def reset(self):
        self.state_future = copy_app_future(self.start_future)
        self.tag = copy_app_future('safe')
        return self.state_future

    def copy(self):
        walker = self.__class__(
                self.context,
                self.state_future,
                )
        walker.start_future = copy_app_future(self.start_future)
        walker.tag_future   = copy_app_future(self.tag_future)
        walker.parameters   = deepcopy(self.parameters)
        return walker

    def save(self, path, require_done=True):
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
    def create_apps(cls, context):
        assert not (cls == BaseWalker) # should never be called directly
        pass
