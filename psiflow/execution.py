from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Callable, Union, Any
import typeguard
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
import logging

from parsl.dataflow.memoization import id_for_memo
from parsl.data_provider.files import File

from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor, \
        WorkQueueExecutor
from parsl.data_provider.files import File
from parsl.config import Config


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


@dataclass
class ModelEvaluationExecution:
    executor: str = 'model'
    device: str = 'cpu'
    ncores: Optional[int] = 1
    dtype: str = 'float32'
    walltime: Optional[int] = None
    memory_per_core: int = 1900
    disk: int = 1000


@dataclass
class ModelTrainingExecution:
    executor: str = 'training'
    device: str = 'cuda'
    ncores: Optional[int] = None
    dtype: str = 'float32'
    walltime: Optional[int] = None
    memory_per_core: int = 1900
    disk: int = 1000


@dataclass
class ReferenceEvaluationExecution:
    executor: str = 'reference'
    device: str = 'cpu'
    mpi_command: Callable = lambda x: f'mpirun -np {x}'
    cp2k_exec: str = 'cp2k.psmp'
    memory_per_core: int = 1900
    disk: int = 1000
    ncores: Optional[int] = 1
    walltime: Optional[int] = None


@typeguard.typechecked
class ExecutionContext:

    def __init__(
            self,
            config: Config,
            path: Union[Path, str],
            enable_logging: bool = True,
            ) -> None:
        self.config = config
        Path.mkdir(Path(path), parents=True, exist_ok=True)
        self.path = Path(path)
        self.executors = {e.label: e for e in config.executors}
        self.execution_definitions = {}
        self._apps = {}
        self.file_index = {}
        assert 'default' in self.executor_labels
        logging.basicConfig(format='%(name)s - %(message)s')
        logging.getLogger('parsl').setLevel(logging.WARNING)

    def __getitem__(
            self,
            container, # container subclass
            ) -> tuple[list, list]:
        assert container in self.execution_definitions.keys()
        return self.execution_definitions[container]

    def define_execution(self, container, *executions) -> None:
        assert container not in self.execution_definitions.keys()
        defined_types = set([type(e) for e in executions])
        assert len(defined_types) == len(executions) # unique execution types
        assert defined_types == container.execution_types
        executions = [deepcopy(e) for e in executions]
        resource_specifications = []
        for execution in executions:
            executor = self.executors[execution.executor]
            # some executors specify the number of cores themselves
            if execution.ncores is None:
                if isinstance(executor, HighThroughputExecutor):
                    execution.ncores = int(executor.cores_per_worker)
                elif isinstance(executor, ThreadPoolExecutor):
                    execution.ncores = 1
                else:
                    raise ValueError('ncores must be specified for WQEX')

            if (execution.walltime is not None):
                if isinstance(executor, WorkQueueExecutor):
                    logger.critical('walltime can only be set when using a '
                            'WorkQueueExecutor; value will be ignored')

            if isinstance(executor, WorkQueueExecutor): # work queue spec 
                resource_specification = {}
                resource_specification['cores'] = execution.ncores
                resource_specification['disk'] = execution.disk
                memory = execution.memory_per_core * execution.ncores
                resource_specification['memory'] = int(memory)
                if execution.device == 'cuda':
                    resource_specification['gpus'] = 1
                if execution.walltime is not None:
                    resource_specification['running_time_min'] = execution.walltime
            else:
                resource_specification = None
            resource_specifications.append(resource_specification)

        definition = (executions, resource_specifications)
        self.execution_definitions[container] = definition
        container.create_apps(self)

    def apps(self, container, app_name: str) -> Callable:
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

    def new_file(self, prefix: str, suffix: str) -> File:
        assert prefix[-1] == '_'
        assert suffix[0]  == '.'
        key = (prefix, suffix)
        if key not in self.file_index.keys():
            self.file_index[key] = 0
        padding = 6
        assert self.file_index[key] < (16 ** padding)
        identifier = '{0:0{1}x}'.format(self.file_index[key], padding)
        self.file_index[key] += 1
        return File(str(self.path / (prefix + identifier + suffix)))

    @property
    def executor_labels(self):
        return list(self.executors.keys())


@typeguard.typechecked
class Container:

    def __init__(self, context: ExecutionContext) -> None:
        self.context = context

    @staticmethod
    def create_apps(context: ExecutionContext):
        raise NotImplementedError


@id_for_memo.register(File)
def id_for_memo_file(file: File, output_ref=False):
    return bytes(file.filepath, 'utf-8')
