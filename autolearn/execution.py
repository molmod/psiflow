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
    #ncores        : int = 1
    dtype         : str = 'float32'


@dataclass(frozen=True)
class CP2KExecutionDefinition:
    executor_label: str  = 'cpu_large'
    ncores        : int  = 1
    command       : str  = 'cp2k.psmp' # default command for CP2K Reference
    mpi           : Optional[Callable] = None # or callable, e.g: mpi(ncores) -> 'mpirun -np {ncores} '
    walltime      : int  = 3600 # timeout in seconds


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
        parsl.load(config)

    def __getitem__(self, definition_class):
        assert definition_class in self.definitions.keys()
        return self.definitions[definition_class]

    def register(self, execution_definition):
        assert execution_definition.executor_label in self.executor_labels
        key = execution_definition.__class__
        assert key not in self.definitions.keys()
        self.definitions[key] = execution_definition
