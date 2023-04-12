from __future__ import annotations # necessary for type-guarding class methods
# see https://stackoverflow.com/questions/59904631/python-class-constants-in-dataclasses
from typing import Optional, Callable, Union, Any, ClassVar, Type
import typeguard
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
import argparse
import logging
import os
import sys
import importlib
import atexit

import parsl
from parsl.launchers.launchers import Launcher
from parsl.executors import HighThroughputExecutor, WorkQueueExecutor
from parsl.providers.base import ExecutionProvider
from parsl.dataflow.memoization import id_for_memo
from parsl.data_provider.files import File
from parsl.config import Config

from psiflow.utils import set_file_logger


logger = logging.getLogger(__name__) # logging per module


class MyWorkQueueExecutor(WorkQueueExecutor):
    
    def _get_launch_command(self, block_id):
        return self.worker_command


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
def get_psiflow_config_from_file(
        path_config: Union[Path, str],
        path_internal: Union[Path, str],
        ) -> tuple[Config, dict]:
    path_config = Path(path_config)
    assert path_config.is_file(), 'cannot find psiflow config at {}'.format(path_config)
    # see https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    spec = importlib.util.spec_from_file_location('module.name', path_config)
    psiflow_config_module = importlib.util.module_from_spec(spec)
    sys.modules['module.name'] = psiflow_config_module
    spec.loader.exec_module(psiflow_config_module)
    return psiflow_config_module.get_config(path_internal)


@typeguard.typechecked
def generate_parsl_config(
        path_internal: Union[Path, str],
        definitions: dict[Any, list[Execution]],
        providers: dict[str, ExecutionProvider],
        use_work_queue: bool = True,
        wq_timeout: int = 120, # in seconds
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
                    working_dir=str(Path(path_internal) / label),
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
                        #'--workdir={}'.format(path_internal),
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
                executor = MyWorkQueueExecutor(
                    label=label,
                    working_dir=str(Path(path_internal) / label),
                    provider=provider,
                    shared_fs=True,
                    autocategory=False,
                    port=0,
                    max_retries=0,
                    worker_options=' '.join(worker_options),
                    )
            else:
                executor = HighThroughputExecutor(
                        address=htex_address,
                        label=label,
                        working_dir=str(Path(path_internal) / label),
                        cores_per_worker=execution.ncores,
                        provider=provider,
                        )
        executors[label] = executor
    config = Config(
            executors=list(executors.values()),
            run_dir=str(path_internal),
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
            definitions: dict[Any, list[Execution]],
            path: Union[Path, str],
            ) -> None:
        self.config = config
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.definitions = definitions
        self._apps = {}
        self.file_index = {}
        parsl.load(config)

    def __getitem__(self, container) -> list[Execution]:
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
            container,
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

    def atexit_cleanup(self) -> None:
        parsl.wait_for_current_tasks()

    @property
    def executor_labels(self):
        return list(self.executors.keys())


class ExecutionContextLoader:
    _context: Optional[ExecutionContext] = None

    @classmethod
    def load(
            cls,
            path_config: Optional[Union[Path, str]] = None,
            path_internal: Optional[Union[Path, str]] = None,
            psiflow_log_level: Union[int, str] = 'INFO',
            parsl_log_level: Union[int, str] = 'INFO',
            ) -> ExecutionContext:
        if cls._context is not None:
            raise RuntimeError('ExecutionContext has already been loaded')
        # convert all paths into absolute paths as this is necssary when using
        # WQ executor with shared_fs=True
        if path_config is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('--psiflow-config', default='', type=str)
            parser.add_argument('--path-internal', default='psiflow_internal', type=str)
            parser.add_argument('--psiflow-log-level', default='INFO', type=str)
            parser.add_argument('--parsl-log-level', default='INFO', type=str)
            args = parser.parse_args()
            path_config = args.psiflow_config
            path_internal = args.path_internal
            psiflow_log_level = args.psiflow_log_level
            parsl_log_level = args.parsl_log_level
        path_internal = Path(path_internal).resolve()
        config, definitions = get_psiflow_config_from_file(
                path_config,
                path_internal,
                )
        if not path_internal.is_dir():
            path_internal.mkdir()
        else:
            assert not any(path_internal.iterdir()), '{} should be empty'.format(str(path_internal))
        path_psiflow_log = Path.cwd().resolve() / 'psiflow.log'
        if path_psiflow_log.is_file():
            path_psiflow_log.unlink()
        set_file_logger(path_psiflow_log, psiflow_log_level)
        parsl.set_file_logger(str(path_internal / 'parsl.log'), parsl_log_level)
        path_context = path_internal / 'context_dir'
        cls._context = ExecutionContext(config, definitions, path_context)
        atexit.register(cls._context.atexit_cleanup)
        return cls._context

    @classmethod
    def context(cls):
        if cls._context is None:
            raise RuntimeError('No ExecutionContext is currently loaded')
        return cls._context


@typeguard.typechecked
class ApptainerLauncher(Launcher):

    def __init__(
        self,
        debug: bool = True,
        apptainer_or_singularity: str = 'apptainer',
        container_tag: Optional[str] = None,
        enable_gpu: Optional[bool] = False,
        cuda_or_rocm: str = 'cuda',
    ) -> None:
        super().__init__(debug=debug)
        self.apptainer_or_singularity = apptainer_or_singularity
        self.container_tag = container_tag
        self.enable_gpu = enable_gpu
        self.cuda_or_rocm = cuda_or_rocm

        self.launch_command = ''
        self.launch_command += apptainer_or_singularity
        self.launch_command += ' exec'
        self.launch_command += ' --no-eval --contain'
        self.launch_command += ' -e --no-mount $HOME/.local' # avoid unwanted python imports from host
        self.launch_command += ' --bind {}'.format(Path.cwd().resolve()) # access to data / internal dir
        self.launch_command += ' -W /tmp' # fix problem with WQ in which workers do not have enough disk space
        self.launch_command += ' --writable-tmpfs' # necessary for wandb
        env  = {}
        keys = ['WANDB_API_KEY']
        for key in keys:
            if key in os.environ.keys():
                env[key] = os.environ[key]
        if 'WANDB_API_KEY' not in env.keys():
            logger.critical('wandb API key not set; please go to wandb.ai/authorize and '
                'set that key in the current environment: export WANDB_API_KEY=<key-from-wandb.ai/authorize>')
        env['PARSL_CORES'] = '${PARSL_CORES}'
        if len(env) > 0:
            self.launch_command += ' --env '
            self.launch_command += ','.join([f'{k}={v}' for k, v in env.items()])
        if enable_gpu:
            if cuda_or_rocm == 'cuda':
                self.launch_command += ' --nv'
            else:
                self.launch_command += ' --rocm'
        self.launch_command += ' docker://ghcr.io/svandenhaute/psiflow:'
        if container_tag is None:
            psiflow_version = importlib.metadata.version('psiflow') 
            pass
        else:
            self.launch_command += container_tag
        self.launch_command += ' /usr/local/bin/_entrypoint.sh '

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        return self.launch_command + "{}".format(command)
