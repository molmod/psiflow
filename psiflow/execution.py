from __future__ import annotations # necessary for type-guarding class methods
# see https://stackoverflow.com/questions/59904631/python-class-constants-in-dataclasses
from typing import Optional, Callable, Union, Any, ClassVar, Type
import typeguard
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
import logging
import os

import parsl
from parsl.executors import HighThroughputExecutor, WorkQueueExecutor
from parsl.providers.base import ExecutionProvider
from parsl.dataflow.memoization import id_for_memo
from parsl.data_provider.files import File
from parsl.config import Config


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
@dataclass(frozen=True, eq=True) # allows checking for equality
class Execution:
    executor: str
    ncores: int
    walltime: Optional[int]

    def generate_parsl_resource_specification(self):
        resource_specification = {}
        resource_specification['cores'] = self.ncores
        resource_specification['disk'] = self.disk
        memory = self.memory_per_core * self.ncores
        resource_specification['memory'] = int(memory)
        if self.device == 'cuda':
            resource_specification['gpus'] = 1
        if self.walltime is not None:
            resource_specification['running_time_min'] = self.walltime
        return resource_specification


@typeguard.typechecked
@dataclass(frozen=True, eq=True) # allows checking for equality
class ModelEvaluationExecution(Execution):
    executor: str = 'model'
    device: str = 'cpu'
    ncores: int = 1
    dtype: str = 'float32'
    walltime: Optional[int] = None
    memory_per_core: int = 1900
    disk: int = 1000


@typeguard.typechecked
@dataclass(frozen=True, eq=True)
class ModelTrainingExecution(Execution):
    device: ClassVar[str] = 'cuda' # fixed
    dtype: ClassVar[str] = 'float32' # fixed
    executor: str = 'training'
    ncores: int = 1
    walltime: Optional[int] = 10
    memory_per_core: int = 1900
    disk: int = 1000


@typeguard.typechecked
@dataclass(frozen=True, eq=True)
class ReferenceEvaluationExecution(Execution):
    executor: str = 'reference'
    device: str = 'cpu'
    mpi_command: Callable = lambda x: f'mpirun -np {x}'
    cp2k_exec: str = 'cp2k.psmp'
    memory_per_core: int = 1900
    disk: int = 1000
    ncores: int = 1
    omp_num_threads: int = 1
    walltime: Optional[int] = None


@typeguard.typechecked
def generate_parsl_config(
        path_parsl_internal: Union[Path, str],
        definitions: dict[Type[Container], list[Execution]],
        providers: dict[str, ExecutionProvider],
        use_work_queue: bool = True,
        wq_timeout: int = 120, # in seconds
        wq_port: int = 9223, # in seconds
        parsl_app_cache: bool = False,
        parsl_retries: int = 0,
        parsl_max_idletime: int = 30, # in seconds
        parsl_strategy: str = 'simple',
        parsl_initialize_logging: bool = True,
        htex_address: Optional[str] = None,
        ) -> Config:
    assert 'default' in providers.keys()
    if htex_address is None: # determine address for htex
        if 'HOSTNAME' in os.environ.keys():
            htex_address = os.environ['HOSTNAME']
        else:
            htex_address = 'localhost'
    executors = {}
    for label, provider in providers.items():
        if label == 'default':
            executor = HighThroughputExecutor(
                    address=htex_address,
                    label='default',
                    working_dir=str(Path(path_parsl_internal) / label),
                    cores_per_worker=1,
                    provider=provider,
                    )
        else:
            execution = None
            for executions in definitions.values():
                for e in executions:
                    if e.executor == label:
                        if execution is not None:
                            assert e == execution
                        else:
                            execution = e
            assert execution is not None
            if use_work_queue:
                worker_options = [
                        '--gpus={}'.format(1 if execution.device == 'cuda' else 0),
                        '--cores={}'.format(execution.ncores),
                        ]
                if hasattr(provider, 'walltime'):
                    walltime_hhmmss = provider.walltime.split(':')
                    assert len(walltime_hhmmss) == 3
                    walltime = 0
                    walltime += 3600 * float(walltime_hhmmss[0])
                    walltime += 60 * float(walltime_hhmmss[1])
                    walltime += float(walltime_hhmmss[2])
                    walltime -= 60 * 4 # add 4 minutes of slack
                    if execution.walltime is not None: # fit at least one app
                        assert 60 * execution.walltime < walltime, ('the '
                                'walltime of your execution definition is '
                                '{}m, which should be less than the total walltime '
                                'available in the corresponding slurm block, '
                                'which is {}'.format(
                                    execution.walltime,
                                    walltime // 60,
                                    ))
                    worker_options.append('--wall-time={}'.format(walltime))
                    worker_options.append('--timeout={}'.format(wq_timeout))
                    worker_options.append('--parent-death')
                executor = WorkQueueExecutor(
                    label=label,
                    working_dir=str(Path(path_parsl_internal) / label),
                    provider=provider,
                    shared_fs=True,
                    autocategory=False,
                    port=wq_port,
                    max_retries=0,
                    worker_options=' '.join(worker_options),
                    )
                wq_port += 1 # use different port for each WQ executor
            else:
                executor = HighThroughputExecutor(
                        address=htex_address,
                        label=label,
                        working_dir=str(Path(path_parsl_internal) / label),
                        cores_per_worker=execution.ncores,
                        provider=provider,
                        )
        executors[label] = executor
    config = Config(
            executors=list(executors.values()),
            run_dir=str(path_parsl_internal),
            usage_tracking=True,
            app_cache=parsl_app_cache,
            retries=parsl_retries,
            initialize_logging=parsl_initialize_logging,
            strategy=parsl_strategy,
            max_idletime=parsl_max_idletime,
            )
    return config


@typeguard.typechecked
class ExecutionContext:

    def __init__(
            self,
            config: Config,
            definitions: dict[Type[Container], list[Execution]],
            path: Union[Path, str],
            ) -> None:
        self.config = config
        Path.mkdir(Path(path), parents=True, exist_ok=True)
        self.path = Path(path)
        self.definitions = definitions
        self._apps = {}
        self.file_index = {}

    def initialize(self):
        parsl.load(self.config)

    def __getitem__(
            self,
            container: Type[Container],
            ) -> list[Execution]:
        assert container in self.definitions.keys(), ('container {}'
                ' has no registered execution definitions with this context. '
                '\navailable definitions are {}'.format(
                    container,
                    list(self.definitions.keys()),
                    ))
        return self.definitions[container]

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
