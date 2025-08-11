from __future__ import annotations  # necessary for type-guarding class methods

import logging
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

# see https://stackoverflow.com/questions/59904631/python-class-constants-in-dataclasses
from typing import Any, Optional, Union, ClassVar, Protocol

import parsl
import psutil
import pytimeparse
import typeguard
import yaml
from parsl.config import Config
from parsl.data_provider.files import File
from parsl.executors import (  # WorkQueueExecutor,
    HighThroughputExecutor,
    ThreadPoolExecutor,
    WorkQueueExecutor,
)
from parsl.executors.base import ParslExecutor
from parsl.launchers import SimpleLauncher, WrappedLauncher
from parsl.launchers.base import Launcher
from parsl.providers import LocalProvider, SlurmProvider
from parsl.providers.base import ExecutionProvider

import psiflow

logger = logging.getLogger(__name__)  # logging per module


EXECUTION_KWARGS = (
    "container_uri",
    "container_engine",
    "container_addopts",
    "container_entrypoint",
)


@dataclass
class ContainerSpec:
    uri: str
    engine: str = "apptainer"
    addopts: str = " --no-eval -e --no-mount home -W /tmp --writable-tmpfs"
    entrypoint: str = "/opt/entry.sh"

    def __post_init__(self):
        assert self.engine in ("apptainer", "singularity")
        assert len(self.uri) > 0

    def launch_command(self, gpu: bool = False) -> str:
        # TODO: pretty sure some of this is overkill
        pwd = Path.cwd().resolve()  # access to data / internal dir
        args = [self.engine, "exec", self.addopts, f"--bind {pwd}"]
        if gpu:
            if "rocm" in self.uri:
                args.append("--rocm")
            else:
                args.append("--nv")
        args += [self.uri, self.entrypoint]
        return " ".join(args)

    @staticmethod
    def from_kwargs(kwargs: dict) -> Optional[ContainerSpec]:
        if "container_uri" not in kwargs:
            return None
        keys = (
            "container_uri",
            "container_engine",
            "container_addopts",
            "container_entrypoint",
        )
        args = [kwargs[key] for key in keys if key in kwargs]
        return ContainerSpec(*args)  # TODO: slightly hacky


@typeguard.typechecked
class ExecutionDefinition:
    def __init__(
        self,
        parsl_provider: ExecutionProvider,
        gpu: bool,
        cores_per_worker: int,
        use_threadpool: bool,
        container: Optional[ContainerSpec],
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.parsl_provider = parsl_provider
        self.gpu = gpu
        self.cores_per_worker = cores_per_worker
        self.use_threadpool = use_threadpool
        self.container = container
        self._max_workers = max_workers

    @property
    def name(self) -> str:
        return self.__class__.__name__

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
        if self._max_workers is not None:
            return self._max_workers
        else:
            return max(1, math.floor(self.cores_available / self.cores_per_worker))

    @property
    def max_runtime(self):
        if type(self.parsl_provider) is SlurmProvider:
            walltime = pytimeparse.parse(self.parsl_provider.walltime)
        else:
            walltime = 1e9
        return walltime

    def create_executor(self, path: Path) -> ParslExecutor:
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
                "--wall-time={}".format(self.max_runtime),
                "--cores={}".format(cores),
            ]
            if self.gpu:
                worker_options.append("--gpus={}".format(self.max_workers))

            # ensure proper scale in
            if getattr(self.parsl_provider, "nodes_per_block", 1) > 1:
                worker_options.append("--idle-timeout={}".format(int(1e6)))
            else:
                worker_options.append("--idle-timeout={}".format(20))

            # only ModelEvaluation / ModelTraining / default_htex run in containers
            if not isinstance(self, ReferenceEvaluation) and self.container:
                prepend = self.container.launch_command()
                worker_executable = f"{prepend} work_queue_worker"
            else:
                worker_executable = "work_queue_worker"

            executor = MyWorkQueueExecutor(
                label=self.name,
                working_dir=str(path / self.name),
                provider=self.parsl_provider,
                shared_fs=True,
                autocategory=False,
                port=0,
                max_retries=1,
                coprocess=False,
                worker_options=" ".join(worker_options),
                worker_executable=worker_executable,
                scaling_cores_per_worker=cores,
            )
        return executor

    @classmethod
    def from_config(
        cls,
        gpu: bool = False,
        cores_per_worker: int = 1,
        use_threadpool: bool = False,
        container: Optional[ContainerSpec] = None,
        **kwargs,
    ):
        # search for any section in the config which defines the Parsl ExecutionProvider
        # if none are found, default to LocalProvider
        # currently only checking for SLURM
        if "slurm" in kwargs:
            provider_cls = SlurmProvider
            provider_kwargs = kwargs.pop("slurm")  # do not allow empty dict
            provider_kwargs["init_blocks"] = 0
            provider_kwargs.setdefault("exclusive", False)
        else:
            provider_cls = LocalProvider  # noqa: F405
            provider_kwargs = kwargs.pop("local", {})

        # if multi-node blocks are requested, make sure we're using SlurmProvider
        if provider_kwargs.get("nodes_per_block", 1) > 1:
            launcher = SlurmLauncher()
        else:
            launcher = SimpleLauncher()

        if container is not None:
            # TODO: why not exactly?
            assert not use_threadpool

        # initialize provider
        parsl_provider = provider_cls(
            launcher=launcher,
            **provider_kwargs,
        )
        return cls(
            parsl_provider=parsl_provider,
            gpu=gpu,
            use_threadpool=use_threadpool,
            container=container,
            cores_per_worker=cores_per_worker,
            **kwargs,
        )


@typeguard.typechecked
class ModelEvaluation(ExecutionDefinition):
    def __init__(
        self,
        max_simulation_time: Optional[float] = None,
        timeout: float = (10 / 60),  # 5 seconds
        env_vars: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if max_simulation_time is not None:
            assert max_simulation_time * 60 < self.max_runtime
        self.max_simulation_time = max_simulation_time
        self.timeout = timeout

        default_env_vars = {
            "OMP_NUM_THREADS": str(self.cores_per_worker),
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "KMP_BLOCKTIME": "1",
            "OMP_PROC_BIND": "false",
            "PYTHONUNBUFFERED": "TRUE",
        }
        if env_vars is None:
            env_vars = default_env_vars
        else:
            default_env_vars.update(env_vars)
            env_vars = default_env_vars
        self.env_vars = env_vars

    def server_command(self):
        command_list = ["psiflow-server"]
        if self.max_simulation_time is not None:
            max_time = 0.9 * (60 * self.max_simulation_time)
            command_list = ["timeout -s 15 {}s".format(max_time), *command_list]
        return " ".join(command_list)

    def client_command(self):
        command_list = ["psiflow-client"]
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
            return {}
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
        env_vars: Optional[dict[str, str]] = None,
        multigpu: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(gpu=gpu, **kwargs)
        assert self.gpu
        if max_training_time is not None:
            assert max_training_time * 60 < self.max_runtime
        self.max_training_time = max_training_time
        self.multigpu = multigpu
        if self.multigpu:
            message = (
                "the max_training_time keyword does not work "
                "in combination with multi-gpu training. Adjust "
                "the maximum number of epochs to control the "
                "duration of training"
            )
            assert self.max_training_time is None, message

        default_env_vars = {
            "OMP_NUM_THREADS": str(self.cores_per_worker),
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "KMP_BLOCKTIME": "1",
            "OMP_PROC_BIND": "spread",  # different from Model Eval
            "PYTHONUNBUFFERED": "TRUE",
        }
        if env_vars is None:
            env_vars = default_env_vars
        else:
            default_env_vars.update(env_vars)
            env_vars = default_env_vars
        self.env_vars = env_vars

    def train_command(self, initialize: bool = False):
        # script = "$(python -c 'import psiflow.models.mace_utils; print(psiflow.models.mace_utils.__file__)')"
        command_list = ["psiflow-mace-train"]
        if (self.max_training_time is not None) and not initialize:
            max_time = 0.9 * (60 * self.max_training_time)
            command_list = ["timeout -s 15 {}s".format(max_time), *command_list]
        return " ".join(command_list)

    def wq_resources(self):
        if self.use_threadpool:
            return {}
        resource_specification = {}

        if self.multigpu:
            nworkers = int(self.cores_available / self.cores_per_worker)
        else:
            nworkers = 1

        resource_specification["gpus"] = nworkers  # one per GPU
        resource_specification["cores"] = self.cores_available
        resource_specification["disk"] = (
            1000 * nworkers
        )  # some random nontrivial amount?
        memory = 1000 * self.cores_available  # similarly rather random
        resource_specification["memory"] = int(memory)
        resource_specification["running_time_min"] = self.max_training_time
        return resource_specification


@typeguard.typechecked
class ReferenceEvaluation(ExecutionDefinition):
    def __init__(
        self,
        spec: ReferenceSpec,
        max_evaluation_time: Optional[float] = None,
        memory_limit: Optional[str] = None,
        **kwargs,
    ) -> None:
        # TODO: how to know which code?
        super().__init__(**kwargs)
        self.spec = spec
        self.max_evaluation_time = max_evaluation_time * 60  # seconds
        if max_evaluation_time:
            assert 0 < max_evaluation_time < self.max_runtime
        self.memory_limit = memory_limit

    def command(self):
        launch_command = self.spec.launch_command()
        kwargs = {k: getattr(self, k) for k in self.spec.reference_args}
        launch_command = launch_command.format(**kwargs)

        if self.container is not None:
            launch_command = f"{self.container.launch_command()} {launch_command}"

        if (max_time := self.max_evaluation_time) is None:
            # leave some slack for startup and cleanup
            max_time = max(0.9 * self.max_runtime, self.max_runtime - 5)
        launch_command = f"timeout -s 9 {max_time}s {launch_command}"

        commands = []
        if self.memory_limit is not None:
            # based on https://stackoverflow.com/a/42865957/2002471
            units = {"KB": 1, "MB": 2**10, "GB": 2**20, "TB": 2**30}

            def parse_size(size):  # TODO: to utils?
                size = size.upper()
                if not re.match(r" ", size):
                    size = re.sub(r"([KMGT]?B)", r" \1", size)
                number, unit = [string.strip() for string in size.split()]
                return int(float(number) * units[unit])

            commands.append(f"ulimit -v {parse_size(self.memory_limit)}")

        # exit code 0 so parsl always thinks bash app succeeded
        return "\n".join([*commands, launch_command, "exit 0"])

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

    @property
    def name(self) -> str:
        return self.spec.name


class ReferenceSpec(Protocol):
    name: ClassVar[str]
    reference_args: ClassVar[tuple[str, ...]]
    mpi_command: str
    mpi_args: tuple[str, ...]
    executable: str

    def launch_command(self) -> str:
        raise NotImplementedError


@dataclass
class CP2KReferenceSpec(ReferenceSpec):
    name = "CP2K"
    reference_args = ("cores_per_worker",)
    mpi_command: str = "mpirun -np {cores_per_worker}"
    mpi_args: tuple[str, ...] = (
        "-ENV OMP_NUM_THREADS=1",
        "--bind-to core",
        "--map-by core",
    )
    executable: str = "cp2k.psmp -i cp2k.inp"

    def launch_command(self):
        # use nprocs = ncores, nthreads = 1
        return " ".join([self.mpi_command, *self.mpi_args, self.executable])


@dataclass
class GPAWReferenceSpec(ReferenceSpec):
    name = "GPAW"
    reference_args = ("cores_per_worker",)
    mpi_command: str = "mpirun -np {cores_per_worker}"
    mpi_args: tuple[str, ...] = (
        "-x OMP_NUM_THREADS=1",
        "--bind-to core",
        "--map-by core",
    )
    executable: str = "gpaw python script_gpaw.py input.json"

    def launch_command(self):
        # use nprocs = ncores, nthreads = 1
        return " ".join([self.mpi_command, *self.mpi_args, self.executable])


@dataclass
class ORCAReferenceSpec(ReferenceSpec):
    name = "ORCA"
    reference_args = ()
    mpi_command: str = ""
    mpi_args: tuple[str, ...] = (
        "-x OMP_NUM_THREADS=1",
        "--bind-to core",
        "--map-by core",
    )
    executable: str = "$(which orca) orca.inp"

    def launch_command(self):
        mpi_str = " ".join(self.mpi_args)
        return f'{self.executable} "{mpi_str}"'


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
        self.lock = Lock()
        parsl.load(config)

    def __enter__(self):
        return parsl.dfk()

    def __exit__(self, exc_type, exc_value, traceback):
        parsl.wait_for_current_tasks()
        logger.debug("Exiting the context manager, calling cleanup for DFK")
        parsl.dfk().cleanup()

    def new_file(self, prefix: str, suffix: str) -> File:
        with self.lock:
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
        parsl_log_level: str = "WARNING",
        usage_tracking: int = 3,
        retries: int = 2,
        strategy: str = "simple",
        max_idletime: float = 20,
        internal_tasks_max_threads: int = 10,
        default_threads: int = 4,
        htex_address: str = "127.0.0.1",
        zip_staging: Optional[bool] = None,
        make_symlinks: bool = True,
        **kwargs,
    ) -> ExecutionContext:
        path = Path.cwd().resolve() / "psiflow_internal"
        psiflow.resolve_and_check(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        parsl.set_file_logger(
            filename=str(path / "parsl.log"),
            name="parsl",
            level=getattr(logging, parsl_log_level),
        )

        # create definitions
        base_container = ContainerSpec.from_kwargs(kwargs)
        kwargs.pop("container_uri", None)
        model_evaluation = ModelEvaluation.from_config(
            container=base_container,
            **kwargs.pop("ModelEvaluation", {}),
        )
        model_training = ModelTraining.from_config(
            container=base_container,
            **kwargs.pop("ModelTraining", {"gpu": True}),  # avoid triggering assertion
        )
        reference_evaluations = []  # reference evaluations might be class specific
        for key in list(kwargs.keys()):
            if key[:4] in REFERENCE_SPECS:  # allow for e.g., CP2K_small
                config = kwargs.pop(key)
                reference_evaluation = ReferenceEvaluation.from_config(
                    spec=init_spec(REFERENCE_SPECS[key[:4]], config),
                    container=ContainerSpec.from_kwargs(kwargs | config),
                    **config,
                )
                reference_evaluations.append(reference_evaluation)
        definitions = [model_evaluation, model_training, *reference_evaluations]

        # create main parsl executors
        executors = [d.create_executor(path=path) for d in definitions]

        # create default executors
        if base_container is not None:
            launcher = WrappedLauncher(prepend=base_container.launch_command())
        else:
            launcher = SimpleLauncher()
        htex = HighThroughputExecutor(
            label="default_htex",
            address=htex_address,
            working_dir=str(path / "default_htex"),
            cores_per_worker=1,
            max_workers_per_node=default_threads,
            cpu_affinity="none",
            provider=LocalProvider(launcher=launcher, init_blocks=0),  # noqa: F405
        )
        threadpool = ThreadPoolExecutor(
            label="default_threads",
            max_threads=default_threads,
            working_dir=str(path),
        )
        executors.extend([htex, threadpool])

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
        context = ExecutionContext(config, definitions, path / "context_dir")

        if make_symlinks:
            src, dest = Path.cwd() / "psiflow_log", path / "parsl.log"
            _create_symlink(src, dest)
            src, dest = (
                Path.cwd() / "psiflow_submit_scripts",
                path / "000" / "submit_scripts",
            )
            _create_symlink(src, dest, is_dir=True)
            src, dest = Path.cwd() / "psiflow_task_logs", path / "000" / "task_logs"
            _create_symlink(src, dest, is_dir=True)

        return context


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
            else:
                assert len(sys.argv) == 2
                path_config = psiflow.resolve_and_check(Path(sys.argv[1]))
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


class SlurmLauncher(Launcher):
    def __init__(self, debug: bool = True, overrides: str = ""):
        super().__init__(debug=debug)
        self.overrides = overrides

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        x = """set -e

NODELIST=$(scontrol show hostnames)
NODE_ARRAY=($NODELIST)
NODE_COUNT=${{#NODE_ARRAY[@]}}
EXPECTED_NODE_COUNT={nodes_per_block}

# Check if the length of NODELIST matches the expected number of nodes
if [ $NODE_COUNT -ne $EXPECTED_NODE_COUNT ]; then
  echo "Error: Expected $EXPECTED_NODE_COUNT nodes, but got $NODE_COUNT nodes."
  exit 1
fi

for NODE in $NODELIST; do
  srun --nodes=1 --ntasks=1 --exact -l {overrides} --nodelist=$NODE {command} &
  if [ $? -ne 0 ]; then
    echo "Command failed on node $NODE"
  fi
done

wait
""".format(
            nodes_per_block=nodes_per_block,
            command=command,
            overrides=self.overrides,
        )
        return x


class MyWorkQueueExecutor(WorkQueueExecutor):
    def _get_launch_command(self, block_id):
        return self.worker_command


def _create_symlink(src: Path, dest: Path, is_dir: bool = False) -> None:
    """Create or replace symbolic link"""
    if src.is_symlink():
        src.unlink()
    if is_dir:
        dest.mkdir(parents=True, exist_ok=True)
    else:
        dest.touch(exist_ok=True)
    src.symlink_to(dest, target_is_directory=is_dir)


REFERENCE_SPECS = {
    "CP2K": CP2KReferenceSpec,
    "GPAW": GPAWReferenceSpec,
    "ORCA": ORCAReferenceSpec,
}


def init_spec(spec_cls: type(ReferenceSpec), kwargs: dict) -> ReferenceSpec:
    keys = ("mpi_command", "mpi_args", "executable")
    cls_kwargs = {k: kwargs[k] for k in keys if k in kwargs}
    return spec_cls(**cls_kwargs)
