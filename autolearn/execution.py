from typing import Optional, Callable
from dataclasses import dataclass

import parsl


@dataclass(frozen=True)
class TrainingExecutionDefinition:
    executor_label: str = 'gpu'
    device        : str = 'cuda'
    dtype         : str = 'float32'


@dataclass(frozen=True)
class ModelExecutionDefinition:
    executor_label: str = 'cpu_small'
    device        : str = 'cpu'
    ncores        : int = 1
    dtype         : str = 'float32'


@dataclass(frozen=True)
class ReferenceExecutionDefinition:
    executor_label: str  = 'cpu_large'
    ncores        : int  = 1
    walltime      : int  = 3600 # timeout in seconds
    mpi           : Optional[Callable] = None # or callable, e.g: mpi(ncores) -> 'mpirun -np {ncores} '
    cp2k_command  : str  = 'cp2k.psmp' # default command for CP2K Reference


@dataclass(frozen=True)
class EMTExecutionDefinition:
    executor_label: str  = 'cpu_small'
    ncores        : int  = 1


class ExecutionContext:

    def __init__(self, config, path):
        self.config = config
        self.path   = path
        self.executor_labels = [e.label for e in config.executors]
        self.definitions = {}
        self._apps = {}
        parsl.load(config)

    def __getitem__(self, definition_class):
        assert definition_class in self.definitions.keys()
        return self.definitions[definition_class]

    def register(self, execution_definition):
        assert execution_definition.executor_label in self.executor_labels
        key = execution_definition.__class__
        assert key not in self.definitions.keys()
        self.definitions[key] = execution_definition

    def apps(self, container, app_name):
        if container not in self._apps.keys():
            container.create_apps(self)
        assert app_name in self._apps[container].keys()
        return self._apps[container][app_name]

    def register_app(self, container, app_name, app):
        if container not in self._apps.keys():
            self._apps[container] = {}
        assert app_name not in self._apps[container].keys()
        self._apps[container][app_name] = app


class Container:

    def __init__(self, context):
        self.context = context

    @staticmethod
    def create_apps(context):
        raise NotImplementedError
