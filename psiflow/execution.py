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
from parsl.launchers import SimpleLauncher, WrappedLauncher
from parsl.providers import LocalProvider, SlurmProvider
from parsl.providers.base import ExecutionProvider

from psiflow.utils import SlurmLauncher, container_launch_command, resolve_and_check

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
class ExecutionDefinition:
    def __init__(
        self,
        parsl_provider: ExecutionProvider,
        gpu: bool,
        cores_per_worker: int,
        use_threadpool: bool,
        worker_prepend: str,
    ) -> None:
        self.parsl_provider = parsl_provider
        self.gpu = gpu
        self.cores_per_worker = cores_per_worker
        self.use_threadpool = use_threadpool
        self.worker_prepend = worker_prepend
        self.name = self.__class__.__name__

    @property
    def cores_available(self):
        if type(self.parsl_provider) is LocalProvider:  # noqa: F405
            cores_available = psutil.cpu_count(logical=False)
        elif type(self.parsl_provider) is SlurmProvider:
            cores_available = self.parsl_provider.cores_per_node
        else:
            cores_available = float("inf")
        return cores_available

    @property
    def max_workers(self):
        return max(1, math.floor(self.cores_available / self.cores_per_worker))

    @property
    def max_runtime(self):
        if type(self.parsl_provider) is SlurmProvider:
            walltime = pytimeparse.parse(self.parsl_provider.walltime)
        else:
            walltime = 1e9
        return walltime

    def create_executor(self, path: Path, **kwargs) -> ParslExecutor:
        if self.use_threadpool:
            executor = ThreadPoolExecutor(
                max_threads=self.cores_per_worker,
                working_dir=str(path),
                label=self.name,
            )
        else:
            cores = self.max_workers * self.cores_per_worker
            worker_options = [
                "--parent-death",
                "--timeout={}".format(30),
                "--wall-time={}".format(self.max_runtime),
                "--cores={}".format(cores),
            ]
            if self.gpu:
                worker_options.append("--gpus={}".format(self.max_workers))

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
                worker_executable="{} work_queue_worker".format(self.worker_prepend),
                scaling_assume_core_slots_per_worker=cores,
            )
        return executor

    @classmethod
    def from_config(
        cls,
        gpu: bool = False,
        cores_per_worker: int = 1,
        use_threadpool: bool = False,
        container: Optional[dict] = None,
        **kwargs,
    ):
        # search for any section in the config which defines the Parsl ExecutionProvider
        # if none are found, default to LocalProvider
        # currently only checking for SLURM
        if "slurm" in kwargs:
            provider_cls = SlurmProvider
            provider_kwargs = kwargs.pop("slurm")  # do not allow empty dict
        else:
            provider_cls = LocalProvider  # noqa: F405
            provider_kwargs = kwargs.pop("local", {})

        # if multi-node blocks are requested, make sure we're using SlurmProvider
        if provider_kwargs.get("nodes_per_block", 1) > 1:
            launcher = SlurmLauncher()
        else:
            launcher = SimpleLauncher()

        if container is not None:
            assert not use_threadpool
            worker_prepend = container_launch_command(gpu=gpu, **container)
        else:
            worker_prepend = ""

        # initialize provider
        parsl_provider = provider_cls(
            launcher=launcher,
            **provider_kwargs,
        )
        return cls(
            parsl_provider=parsl_provider,
            gpu=gpu,
            use_threadpool=use_threadpool,
            worker_prepend=worker_prepend,
            cores_per_worker=cores_per_worker,
            **kwargs,
        )


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
        if self.gpu:
            resource_specification["gpus"] = nclients
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
            return {}
        resource_specification = {}
        resource_specification["cores"] = self.cores_per_worker
        resource_specification["disk"] = 1000  # some random nontrivial amount?
        memory = 2000 * self.cores_per_worker  # similarly rather random
        resource_specification["memory"] = int(memory)
        resource_specification["running_time_min"] = self.max_training_time
        if self.gpu:
            resource_specification["gpus"] = 1
        return resource_specification


@typeguard.typechecked
class ReferenceEvaluation(ExecutionDefinition):
    def __init__(
        self,
        name: str,
        launch_command: Optional[str] = None,
        max_evaluation_time: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = name  # override name
        if max_evaluation_time is None:
            max_evaluation_time = self.max_runtime / 60
        assert max_evaluation_time * 60 <= self.max_runtime
        self.max_evaluation_time = max_evaluation_time

        if launch_command is None:
            launch_command = self.default_launch_command
        self.launch_command = launch_command

    @property
    def default_launch_command(self):
        if self.name.startswith("CP2K"):
            ranks = self.cores_per_worker  # use nprocs = ncores, nthreads = 1
            mpi_command = "mpirun -np {} ".format(ranks)
            mpi_command += "-x OMP_NUM_THREADS=1 "  # cp2k runs best with these settings
            mpi_command += "--bind-to core --map-by core"  # set explicitly
            command = "cp2k.psmp -i cp2k.inp"
            return " ".join([mpi_command, command])
        if self.name.startswith("GPAW"):
            ranks = self.cores_per_worker  # use nprocs = ncores, nthreads = 1
            mpi_command = "mpirun -np {} ".format(ranks)
            mpi_command += "-x OMP_NUM_THREADS=1 "  # cp2k runs best with these settings
            mpi_command += "--bind-to core --map-by core"  # set explicitly
            script = "$(python -c 'import psiflow.reference.gpaw_; print(psiflow.reference.gpaw_.__file__)')"
            return " ".join([mpi_command, "gpaw", "python", script])

    def command(self):
        max_time = 0.9 * (60 * self.max_evaluation_time)
        command = " ".join(
            [
                "timeout -s 9 {}s".format(max_time),
                self.launch_command,
                "|| true",
            ]
        )
        return command

    # def gpaw_command(self):
    #    if self.max_evaluation_time is not None:
    #        max_time = 0.9 * (60 * self.max_evaluation_time)

    def wq_resources(self):
        if self.use_threadpool:
            return {}
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

    def __enter__(self):
        return parsl.dfk()

    def __exit__(self, exc_type, exc_value, traceback):
        parsl.wait_for_current_tasks()
        logger.debug("Exiting the context manager, calling cleanup for DFK")
        parsl.dfk().cleanup()

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
        parsl_log_level: str = "WARNING",
        usage_tracking: bool = True,
        retries: int = 0,
        strategy: str = "simple",
        max_idletime: float = 20,
        internal_tasks_max_threads: int = 10,
        default_threads: int = 4,
        htex_address: Optional[str] = None,
        zip_staging: Optional[bool] = None,
        container_uri: Optional[str] = None,
        container_engine: str = "apptainer",
        container_addopts: str = " --no-eval -e --no-mount home -W /tmp --writable-tmpfs",
        container_entrypoint: str = "/opt/entry.sh",
        **kwargs,
    ) -> ExecutionContext:
        if path is None:
            path = Path.cwd().resolve() / "psiflow_internal"
        resolve_and_check(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        parsl.set_file_logger(
            filename=str(path / "parsl.log"),
            name="parsl",
            level=getattr(logging, parsl_log_level),
        )
        if container_uri is not None:
            container = {
                "uri": container_uri,
                "engine": container_engine,
                "addopts": container_addopts,
                "entrypoint": container_entrypoint,
            }
        else:
            container = None

        # create definitions
        model_evaluation = ModelEvaluation.from_config(
            container=container,
            **kwargs.pop("ModelEvaluation", {}),
        )
        model_training = ModelTraining.from_config(
            container=container,
            **kwargs.pop("ModelTraining", {}),
        )
        reference_evaluations = []  # reference evaluations might be class specific
        for key in list(kwargs.keys()):
            if key[:4] in ["CP2K", "GPAW"]:
                config = kwargs.pop(key)
                reference_evaluation = ReferenceEvaluation.from_config(
                    name=key,
                    **config,
                )
                reference_evaluations.append(reference_evaluation)
        definitions = [model_evaluation, model_training, *reference_evaluations]

        # create main parsl executors
        executors = [d.create_executor(path=path) for d in definitions]

        # create default executors
        if container is not None:
            launcher = WrappedLauncher(prepend=container_launch_command(**container))
        else:
            launcher = SimpleLauncher()
        if htex_address is None:
            htex_address = address_by_hostname()
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
        # if zip_staging:

        #    def zip_uri(base, task_record, err_or_out):
        #        zip_path = base / "base.zip"
        #        file = f"{task_record['func_name']}.{task_record['id']}.{task_record['try_id']}.{err_or_out}"
        #        return File(f"zip:{zip_path}/{file}")

        #    std_autopath = partial(zip_uri, path)
        # else:
        #    std_autopath = None
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
            # std_autopath=std_autopath,
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
                }
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
        return cls._context

    @classmethod
    def context(cls):
        if cls._context is None:
            raise RuntimeError("No ExecutionContext is currently loaded")
        return cls._context

    @classmethod
    def wait(cls):
        parsl.wait_for_current_tasks()
