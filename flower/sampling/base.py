from dataclasses import dataclass
from copy import deepcopy

from parsl.app.app import python_app

from flower.execution import ModelExecutionDefinition, Container
from flower.utils import copy_app_future, unpack_i


def safe_return(state, start, tag):
    if tag == 'unsafe':
        return start
    else:
        return state


def is_reset(state, start):
    if state == start: # positions, numbers, cell, pbc
        return True
    else:
        return False


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
        self.tag_future   = 'safe'

        # parameters
        self.parameters = self.parameters_cls(**kwargs)

    def propagate(self, safe_return=False, **kwargs):
        app = self.context.apps(self.__class__, 'propagate')
        result = app(
                self.state_future,
                self.parameters,
                **kwargs, # Model or Bias instance
                )
        self.state_future = unpack_i(result, 0)
        self.tag_future   = unpack_i(result, 1)
        if safe_return: # only return state if safe, else return start
            # this does NOT reset the walker!
            return self.context.apps(self.__class__, 'safe_return')(
                    self.state_future,
                    self.start_future,
                    self.tag_future,
                    )
        else:
            return self.state_future

    def reset_if_unsafe(self):
        app = self.context.apps(self.__class__, 'safe_return')
        self.state_future = app(
                self.state_future,
                self.start_future,
                self.tag_future,
                )

    def is_reset(self):
        app = self.context.apps(self.__class__, 'is_reset')
        return app(self.state_future, self.start_future)

    def copy(self):
        walker = self.__class__(
                self.context,
                self.state_future,
                )
        walker.start_future = copy_app_future(self.start_future)
        walker.tag_future   = copy_app_future(self.tag_future) # possibly unsafe
        walker.parameters   = deepcopy(self.parameters)
        return walker

    @classmethod
    def create_apps(cls, context):
        assert not (cls == BaseWalker) # should never be called directly
        executor_label = context[ModelExecutionDefinition].executor_label

        app_safe_return = python_app(safe_return, executors=[executor_label])
        context.register_app(cls, 'safe_return', app_safe_return)

        app_is_reset = python_app(is_reset, executors=[executor_label])
        context.register_app(cls, 'is_reset', app_is_reset)
