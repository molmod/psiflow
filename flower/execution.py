from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Callable, Union
import typeguard
from dataclasses import dataclass
from pathlib import Path

from parsl.executors import HighThroughputExecutor
from parsl.config import Config


class ExecutionDefinition:
    pass


@dataclass(frozen=True)
class TrainingExecutionDefinition(ExecutionDefinition):
    label : str = 'training'
    device: str = 'cuda'
    dtype : str = 'float32'


@dataclass(frozen=True)
class ModelExecutionDefinition(ExecutionDefinition):
    label : str = 'model'
    device: str = 'cpu'
    ncores: int = 1
    dtype : str = 'float32'


@dataclass(frozen=True)
class ReferenceExecutionDefinition(ExecutionDefinition):
    device     : str = 'cpu'
    label      : str = 'reference'
    ncores     : int = 1
    mpi_command: Optional[Callable] = lambda x: f'mpirun -np {x} --oversubscribe'
    cp2k_exec  : str = 'cp2k.psmp' # default command for CP2K Reference
    time_per_singlepoint: float = 20


@typeguard.typechecked
class ExecutionContext:

    def __init__(self, config: Config, path: Union[Path, str]) -> None:
        self.config = config
        Path.mkdir(Path(path), parents=True, exist_ok=True)
        self.path = path
        self.executor_labels = [e.label for e in config.executors]
        self.execution_definitions = {}
        self._apps = {}
        assert 'default' in self.executor_labels

    def __getitem__(
            self,
            definition_class: type[ExecutionDefinition],
            ) -> ExecutionDefinition:
        assert definition_class in self.execution_definitions.keys()
        return self.execution_definitions[definition_class]

    def register(self, execution: ExecutionDefinition) -> None:
        assert execution.label in self.executor_labels
        key = execution.__class__
        if execution.device == 'cpu': # check whether cores are available
            found = False
            for executor in self.config.executors:
                if executor.label == execution.label:
                    if type(executor) == HighThroughputExecutor:
                        assert executor.cores_per_worker == execution.ncores
        assert key not in self.execution_definitions.keys()
        self.execution_definitions[key] = execution

    def apps(self, container, app_name: str) -> Callable:
        if container not in self._apps.keys():
            container.create_apps(self)
        assert app_name in self._apps[container].keys()
        return self._apps[container][app_name]

    def register_app(
            self,
            container, # type hints fail to allow Container subclasses?
            app_name: str,
            app: Callable,
            ) -> None:
        if container not in self._apps.keys():
            self._apps[container] = {}
        assert app_name not in self._apps[container].keys()
        self._apps[container][app_name] = app


@typeguard.typechecked
class Container:

    def __init__(self, context: ExecutionContext) -> None:
        self.context = context

    @staticmethod
    def create_apps(context: ExecutionContext):
        raise NotImplementedError
