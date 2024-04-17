from __future__ import annotations  # necessary for type-guarding class methods

import logging
import math
import shutil
import sys
from pathlib import Path

# see https://stackoverflow.com/questions/59904631/python-class-constants-in-dataclasses
from typing import Any, Optional, Union

import parsl
import psutil
import pytimeparse
import typeguard
import yaml
from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.data_provider.files import File
from parsl.executors import (
    HighThroughputExecutor,
    ThreadPoolExecutor,
    WorkQueueExecutor,
)
from parsl.executors.base import ParslExecutor
from parsl.launchers.launchers import SimpleLauncher, SrunLauncher
from parsl.providers import LocalProvider, SlurmProvider
from parsl.providers.base import ExecutionProvider

from psiflow.parsl_utils import ContainerizedLauncher, ContainerizedSrunLauncher
from psiflow.utils import resolve_and_check, set_logger

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
class ExecutionDefinition:
    def __init__(
        self,
        parsl_provider: ExecutionProvider,
        gpu: bool = False,
        cores_per_worker: int = 1,
        use_threadpool: bool = False,
        cpu_affinity: str = "block",
    ) -> None:
        self.parsl_provider = parsl_provider
        self.gpu = gpu
        self.cores_per_worker = cores_per_worker
        self.use_threadpool = use_threadpool
        self.cpu_affinity = cpu_affinity
        self.name = self.__class__.__name__

    @property
    def max_workers(self):
        if type(self.parsl_provider) is LocalProvider:  # noqa: F405
            cores_available = psutil.cpu_count(logical=False)
        elif type(self.parsl_provider) is SlurmProvider:
            cores_available = self.parsl_provider.cores_per_node
        else:
            cores_available = float("inf")
        return max(1, math.floor(cores_available / self.cores_per_worker))

    @property
    def max_runtime(self):
        if type(self.parsl_provider) is SlurmProvider:
            walltime = pytimeparse.parse(self.parsl_provider.walltime)
        else:
            walltime = 1e9
        return walltime

    def create_executor(
        self, path: Path, htex_address: Optional[str] = None, **kwargs
    ) -> ParslExecutor:
        if self.use_threadpool:
            executor = ThreadPoolExecutor(
                max_threads=self.cores_per_worker,
                working_dir=str(path),
                label=self.name,
            )
        else:
            worker_options = [
                "--parent-death",
                "--timeout={}".format(30),
                "--wall-time={}".format(self.max_runtime),
            ]
            executor = WorkQueueExecutor(
                label=self.name,
                working_dir=str(path / self.name),
                provider=self.parsl_provider,
                shared_fs=True,
                autocategory=False,
                port=0,
                max_retries=0,
                coprocess=False,
                worker_options=" ".join(worker_options),
            )
        return executor

    @classmethod
    def from_config(
        cls,
        config_dict: dict,
        container: Optional[dict] = None,
    ):
        if "container" in config_dict:
            container = config_dict.pop("container")  # only used once

        # search for any section in the config which defines the Parsl ExecutionProvider
        # if none are found, default to LocalProvider
        provider_keys = list(filter(lambda k: "Provider" in k, config_dict.keys()))
        if len(provider_keys) == 0:
            provider_cls = LocalProvider  # noqa: F405
            provider_dict = {}
        elif len(provider_keys) == 1:
            provider_cls = getattr(sys.modules[__name__], provider_keys[0])
            provider_dict = config_dict.pop(provider_keys[0])

        # if multi-node blocks are requested, make sure we're using SlurmProvider
        if provider_dict.get("nodes_per_block", 1) > 1:
            assert (
                provider_keys[0] == "SlurmProvider"
            ), "multi-node blocks only supported for SLURM"
            if container is not None:
                launcher = ContainerizedSrunLauncher(
                    **container,
                    enable_gpu=config_dict.get("gpu", False),
                )
            else:
                launcher = SrunLauncher()
        else:
            launcher = SimpleLauncher()

        # initialize provider
        parsl_provider = provider_cls(launcher=launcher, **provider_dict)
        return cls(parsl_provider=parsl_provider, **config_dict)


@typeguard.typechecked
class ModelEvaluation(ExecutionDefinition):
    def __init__(
        self,
        max_simulation_time: Optional[float] = None,
        timeout: float = (10 / 60),  # 5 seconds
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if max_simulation_time is not None:
            assert max_simulation_time * 60 < self.max_runtime
        self.max_simulation_time = max_simulation_time
        self.timeout = timeout

    def server_command(self):
        script = "$(python -c 'import psiflow.tools.server; print(psiflow.tools.server.__file__)')"
        command_list = ["python", "-u", script]
        if self.max_simulation_time is not None:
            max_time = 0.9 * (60 * self.max_simulation_time)
            command_list = ["timeout -s 15 {}s".format(max_time), *command_list]
        return " ".join(command_list)

    def client_command(self):
        script = "$(python -c 'import psiflow.tools.client; print(psiflow.tools.client.__file__)')"
        command_list = ["python", "-u", script]
        return " ".join(command_list)

    def get_client_args(
        self,
        hamiltonian_name: str,
        nwalkers: int,
        motion: str,
    ) -> list[str]:
        if "MACE" in hamiltonian_name:
            if motion in ["minimize", "vibrations"]:
                dtype = "float64"
            else:
                dtype = "float32"
            nclients = min(nwalkers, self.max_workers)
            if self.gpu:
                template = "--dtype={} --device=cuda:{}"
                args = [template.format(dtype, i) for i in range(nclients)]
            else:
                template = "--dtype={} --device=cpu"
                args = [template.format(dtype) for i in range(nclients)]
            return args
        else:
            return [""]

    def wq_resources(self, nwalkers):
        if self.use_threadpool:
            return None
        nclients = min(nwalkers, self.max_workers)
        resource_specification = {}
        resource_specification["cores"] = nclients * self.cores_per_worker
        resource_specification["disk"] = 1000  # some random nontrivial amount?
        memory = 2000 * self.cores_per_worker  # similarly rather random
        resource_specification["memory"] = int(memory)
        resource_specification["running_time_min"] = self.max_simulation_time
        return resource_specification


@typeguard.typechecked
class ModelTraining(ExecutionDefinition):
    def __init__(
        self,
        gpu=True,
        max_training_time: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(gpu=gpu, **kwargs)
        if max_training_time is not None:
            assert max_training_time * 60 < self.max_runtime
        self.max_training_time = max_training_time

    def train_command(self, initialize: bool = False):
        script = "$(python -c 'import psiflow.models.mace_utils; print(psiflow.models.mace_utils.__file__)')"
        command_list = ["python", script]
        if (self.max_training_time is not None) and not initialize:
            max_time = 0.9 * (60 * self.max_training_time)
            command_list = ["timeout -s 15 {}s".format(max_time), *command_list]
        return " ".join(command_list)

    def wq_resources(self):
        if self.use_threadpool:
            return None
        resource_specification = {}
        resource_specification["cores"] = self.cores_per_worker
        resource_specification["disk"] = 1000  # some random nontrivial amount?
        memory = 2000 * self.cores_per_worker  # similarly rather random
        resource_specification["memory"] = int(memory)
        resource_specification["running_time_min"] = self.max_training_time
        resource_specification["gpus"] = 1
        return resource_specification


@typeguard.typechecked
class ReferenceEvaluation(ExecutionDefinition):
    def __init__(
        self,
        name: Optional[str] = None,
        mpi_command: Optional[str] = None,
        cpu_affinity: str = "none",  # better default for cp2k
        max_evaluation_time: Optional[float] = None,
        cp2k_executable: str = "cp2k.psmp",
        **kwargs,
    ) -> None:
        super().__init__(cpu_affinity=cpu_affinity, **kwargs)
        if max_evaluation_time is not None:
            assert max_evaluation_time * 60 < self.max_runtime
        self.max_evaluation_time = max_evaluation_time
        self.cp2k_executable = cp2k_executable
        if mpi_command is None:  # parse
            ranks = self.cores_per_worker  # use nprocs = ncores, nthreads = 1
            mpi_command = "mpirun -np {} -bind-to core -rmk user -launcher fork -x OMP_NUM_THREADS=1".format(
                ranks
            )
        self.mpi_command = mpi_command
        if name is not None:
            self.name = name  # if not None, the name of the reference class

    def cp2k_command(self):
        command = " ".join(
            [
                self.mpi_command,
                self.cp2k_executable,
                "-i cp2k.inp",
            ]
        )
        if self.max_evaluation_time is not None:
            max_time = 60 * self.max_evaluation_time
            command = " ".join(
                [
                    "timeout -s 9 {}s".format(max_time),
                    command,
                    "|| true",
                ]
            )
        return command

    def wq_resources(self):
        if self.use_threadpool:
            return None
        resource_specification = {}
        resource_specification["cores"] = self.cores_per_worker
        resource_specification["disk"] = 1000  # some random nontrivial amount?
        memory = 2000 * self.cores_per_worker  # similarly rather random
        resource_specification["memory"] = int(memory)
        resource_specification["running_time_min"] = self.max_evaluation_time
        return resource_specification


@typeguard.typechecked
class ExecutionContext:
    """
    Psiflow centralizes all execution-level configuration options using an ExecutionContext.
    It forwards infrastructure-specific options within Parsl, such as the
    requested number of nodes per SLURM job or the specific Google Cloud instance to be use,
    to training, sampling, and QM evaluation operations to ensure they proceed as requested.
    Effectively, the ExecutionContext hides all details of the execution infrastructure and
    exposes simple and platform-agnostic resources which may be used by training, sampling,
    and QM evaluation apps. As such, we ensure that execution-side details are strictly
    separated from the definition of the computational graph itself.
    For more information, check out the psiflow documentation regarding execution.

    """

    def __init__(
        self,
        config: Config,
        definitions: list[ExecutionDefinition],
        path: Union[Path, str],
    ) -> None:
        self.config = config
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.definitions = {d.name: d for d in definitions}
        assert len(self.definitions) == len(definitions)
        self.file_index = {}
        parsl.load(config)

    def new_file(self, prefix: str, suffix: str) -> File:
        assert prefix[-1] == "_"
        assert suffix[0] == "."
        key = (prefix, suffix)
        if key not in self.file_index.keys():
            self.file_index[key] = 0
        padding = 6
        assert self.file_index[key] < (16**padding)
        identifier = "{0:0{1}x}".format(self.file_index[key], padding)
        self.file_index[key] += 1
        return File(str(self.path / (prefix + identifier + suffix)))

    @classmethod
    def from_config(
        cls,
        path: Optional[Union[str, Path]] = None,
        parsl_log_level: str = "INFO",
        psiflow_log_level: str = "INFO",
        usage_tracking: bool = True,
        retries: int = 2,
        strategy: str = "simple",
        max_idletime: float = 20,
        internal_tasks_max_threads: int = 10,
        default_threads: int = 1,
        htex_address: Optional[str] = None,
        container: Optional[dict] = None,
        **kwargs,
    ) -> ExecutionContext:
        if path is None:
            path = Path.cwd().resolve() / "psiflow_internal"
        resolve_and_check(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        parsl.set_file_logger(
            str(path / "parsl.log"),
            "parsl",
            getattr(logging, parsl_log_level),
        )
        set_logger(psiflow_log_level)

        # create definitions
        model_evaluation = ModelEvaluation.from_config(
            config_dict=kwargs.pop("ModelEvaluation", {}),
            container=container,
        )
        model_training = ModelTraining.from_config(
            config_dict=kwargs.pop("ModelTraining", {}),
            container=container,
        )
        reference_evaluations = []  # reference evaluations might be class specific
        for key in list(kwargs.keys()):
            if key.endswith("ReferenceEvaluation"):
                config_dict = kwargs.pop(key)
                config_dict["name"] = key
                reference_evaluation = ReferenceEvaluation.from_config(
                    config_dict=config_dict,
                    container=container,
                )
                reference_evaluations.append(reference_evaluation)
        if len(reference_evaluations) == 0:
            reference_evaluation = ReferenceEvaluation.from_config(
                config_dict={},
                container=container,
            )
            reference_evaluations.append(reference_evaluation)
        definitions = [model_evaluation, model_training, *reference_evaluations]

        # create main parsl executors
        if htex_address is None:
            htex_address = address_by_hostname()
        executors = [d.create_executor(path=path) for d in definitions]

        # create default executors
        if container is not None:
            launcher = ContainerizedLauncher(**container)
        else:
            launcher = SimpleLauncher()
        htex = HighThroughputExecutor(
            label="default_htex",
            address=htex_address,
            working_dir=str(path / "default_htex"),
            cores_per_worker=1,
            max_workers=1,
            provider=LocalProvider(launcher=launcher),  # noqa: F405
        )
        executors.append(htex)
        threadpool = ThreadPoolExecutor(
            label="default_threads",
            max_threads=default_threads,
            working_dir=str(path),
        )
        executors.append(threadpool)

        # remove additional kwargs
        config = Config(
            executors=executors,
            run_dir=str(path),
            initialize_logging=False,
            app_cache=False,
            usage_tracking=usage_tracking,
            retries=retries,
            strategy=strategy,
            max_idletime=max_idletime,
            internal_tasks_max_threads=internal_tasks_max_threads,
        )
        return ExecutionContext(config, definitions, path / "context_dir")


class ExecutionContextLoader:
    _context: Optional[ExecutionContext] = None

    @classmethod
    def load(
        cls,
        psiflow_config: Optional[dict[str, Any]] = None,
    ) -> ExecutionContext:
        if cls._context is not None:
            raise RuntimeError("ExecutionContext has already been loaded")
        if psiflow_config is None:  # assume yaml is passed as argument
            if len(sys.argv) == 1:  # no config passed, use threadpools:
                psiflow_config = {
                    "ModelEvaluation": {
                        "gpu": False,
                        "use_threadpool": True,
                    },
                    "ModelTraining": {
                        "gpu": True,
                        "use_threadpool": True,
                    },
                    "ReferenceEvaluation": {
                        "max_evaluation_time": 1e9,
                        "mpi_command": "mpirun -np 1",
                        "use_threadpool": True,
                    },
                }
                psiflow_config = {}
                path = Path.cwd() / ".psiflow_internal"
                if path.exists():
                    shutil.rmtree(path)
                psiflow_config["path"] = path
            else:
                assert len(sys.argv) == 2
                path_config = resolve_and_check(Path(sys.argv[1]))
                assert path_config.exists()
                assert path_config.suffix in [".yaml", ".yml"], (
                    "the execution configuration needs to be specified"
                    " as a YAML file, but got {}".format(path_config)
                )
                with open(path_config, "r") as f:
                    psiflow_config = yaml.safe_load(f)
        cls._context = ExecutionContext.from_config(**psiflow_config)

    @classmethod
    def context(cls):
        if cls._context is None:
            raise RuntimeError("No ExecutionContext is currently loaded")
        return cls._context

    @classmethod
    def wait(cls):
        parsl.wait_for_current_tasks()
