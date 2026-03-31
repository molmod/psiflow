import logging
import shutil
import sys
import subprocess
import inspect
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union, ClassVar, Protocol, Sequence, TypeVar

import parsl
import psutil
import yaml
from parsl.config import Config
from parsl.data_provider.files import File
from parsl.executors import (
    HighThroughputExecutor,
    ThreadPoolExecutor,
    WorkQueueExecutor,
)
from parsl.executors.base import ParslExecutor
from parsl.launchers import SimpleLauncher, WrappedLauncher
from parsl.launchers.base import Launcher
from parsl.providers import LocalProvider, SlurmProvider
from parsl.providers.base import ExecutionProvider

from psiflow.utils.config import (
    PSIFLOW_INTERNAL,
    PARSL_LOGFILE,
    PSIFLOW_LOGFILE,
    DEFAULT_CONFIG,
    CONTEXT_DIR,
)
from psiflow.utils.logging import setup_logging
from psiflow.utils.wq import register_definition
from psiflow.utils.apps import create_bash_template
from psiflow.utils.parse import str_to_timedelta


logger = logging.getLogger(__name__)  # logging per module


class ConfigurationError(ValueError):
    pass  # some global psiflow configuration option does not make sense


def ensure(
    *conditions: bool,
    msg: str = "Whoopsie",
    msgs: Sequence[str] = (),
    template: str = "{}",
) -> None:
    """Small helper function to replace 'assert' statements"""
    if all(conditions):
        return
    if len(msgs) == 0:
        raise ConfigurationError(msg)
    msg = msgs[conditions.index(False)]
    raise ConfigurationError(template.format(msg))


@dataclass
class ContainerSpec:
    """Controls container configuration"""

    uri: str
    engine: str = "apptainer"
    addopts: str = " --no-eval -e --no-mount home -W /tmp --writable-tmpfs"
    gpu_flavour: str | None = None

    def __post_init__(self):
        ensure(
            self.engine in ("apptainer", "singularity"),
            len(self.uri) > 0,
            self.gpu_flavour in ("cuda", "rocm", None),
            msg="Invalid container configuration",
        )

    def launch_command(self) -> str:
        pwd = Path.cwd().resolve()  # access to data / internal dir
        gpu = {"cuda": "--nv", "rocm": "--rocm"}.get(self.gpu_flavour, "")
        return f"{self.engine} run {self.addopts} {gpu} --bind {pwd} {self.uri}"


class ReferenceSpec(Protocol):
    """Defines default options for Reference implementations"""

    name: ClassVar[str]
    reference_args: ClassVar[tuple[str, ...]]
    mpi_command: str
    mpi_args: Sequence[str]
    executable: str

    def launch_command(self) -> str:
        raise NotImplementedError


@dataclass
class CP2KReferenceSpec(ReferenceSpec):
    name = "CP2K"
    reference_args = ("cores_per_task",)
    mpi_command: str = "mpiexec -n {cores_per_task}"
    mpi_args: Sequence[str] = (
        "-genv OMP_NUM_THREADS=1",
        "--bind-to core",
        "--map-by core",
    )
    executable: str = "cp2k.psmp -i cp2k.inp"

    def launch_command(self):
        return " ".join([self.mpi_command, *self.mpi_args, self.executable])


@dataclass
class GPAWReferenceSpec(ReferenceSpec):
    name = "GPAW"
    reference_args = ("cores_per_task",)
    mpi_command: str = "mpirun -np {cores_per_task}"
    mpi_args: Sequence[str] = (
        "-x OMP_NUM_THREADS=1",
        "--bind-to core",
        "--map-by core",
    )
    executable: str = "gpaw python script_gpaw.py input.json"

    def launch_command(self):
        return " ".join([self.mpi_command, *self.mpi_args, self.executable])


@dataclass
class ORCAReferenceSpec(ReferenceSpec):
    name = "ORCA"
    reference_args = ()
    mpi_command: str = ""
    mpi_args: Sequence[str] = (
        "-x OMP_NUM_THREADS=1",
        "--bind-to core",
        "--map-by core",
    )
    executable: str = "$(which orca) orca.inp"

    def launch_command(self):
        mpi_str = " ".join(self.mpi_args)
        return f'{self.executable} "{mpi_str}"'


REFERENCE_SPECS = {
    "CP2K": CP2KReferenceSpec,
    "GPAW": GPAWReferenceSpec,
    "ORCA": ORCAReferenceSpec,
}


def make_slurm_provider(kwargs: dict) -> tuple[SlurmProvider, dict]:
    defaults = {"init_blocks": 0, "exclusive": False}
    required = ("cores_per_node", "walltime")
    kwargs = defaults | kwargs
    ensure(all(key in kwargs for key in required))
    provider = SlurmProvider(**kwargs)  # does not configure Launcher
    resources = {
        "nodes": provider.nodes_per_block,
        "cores": provider.cores_per_node,
        "memory": provider.mem_per_node or float("inf"),
        "gpus": provider.gpus_per_node or 0,
        "lifetime": str_to_timedelta(provider.walltime).seconds,
    }
    return provider, resources


def make_local_provider(kwargs: dict) -> tuple[LocalProvider, dict]:
    resources = {
        "nodes": 1,
        "cores": kwargs.get("cores", psutil.cpu_count(logical=False)),
        "memory": kwargs.get(
            "memory", psutil.virtual_memory().available / 1e9
        ),  # TODO: available?
        "lifetime": float("inf"),
    }
    if "gpus" in kwargs:
        resources["gpus"] = kwargs["gpus"]
    else:
        out = ""
        try:
            out = subprocess.check_output(
                "nvidia-smi -L || amd-smi list",
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pass  # nvidia-sm and amd-smi not found  TODO: not properly tested
        resources["gpus"] = out.count("\n")
    provider = LocalProvider(init_blocks=0)
    return provider, resources


def make_default_executors(
    max_workers: int, path: Path, container: ContainerSpec
) -> tuple[HighThroughputExecutor, ThreadPoolExecutor]:
    """Construct executors for internal app handling"""
    launcher = SimpleLauncher()
    if container is not None:
        launcher = WrappedLauncher(prepend=container.launch_command())

    htex = HighThroughputExecutor(
        label="default_htex",
        working_dir=str(path / "default_htex"),
        cores_per_worker=1,
        max_workers_per_node=max_workers,
        cpu_affinity="none",
        provider=LocalProvider(launcher=launcher, init_blocks=0),
    )
    threadpool = ThreadPoolExecutor(
        label="default_threads",
        max_threads=max_workers,
        working_dir=str(path),
    )
    return htex, threadpool


class ExecutionDefinition:
    def __init__(
        self,
        provider: ExecutionProvider | None,
        executor_type: str,
        executor_kwargs: dict,
        resources: dict,
        container: Optional[ContainerSpec],
        max_runtime: str | None = None,
        env_vars: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        self.provider = provider
        self.executor_type = executor_type
        self.kwargs = executor_kwargs
        self.resources = resources  # compute per node
        self.container = container

        if self.use_gpu:
            ensure(
                resources["gpus"] > 0, msg="GPU usage requested but no GPUs available"
            )
            ensure(
                container is None or container.gpu_flavour is not None,
                msg="Provide container 'gpu_flavour' to choose between CUDA and ROCM",
            )

        if self.executor_type == "workqueue":
            # WQ-specific checks
            ensure(
                self.kwargs["gpus_per_task"] <= resources["gpus"],
                self.kwargs["cores_per_task"] <= resources["cores"],
                self.kwargs["mem_per_task"] <= resources["memory"],
                msgs=["GPUs", "cores", "memory"],
                template="Apps will request more {} than available per Parsl block",
            )

        # how long can individual tasks run (in seconds)
        if max_runtime is None:
            # allow some margin for task cleanup
            max_runtime = max(0.9 * self.lifetime, self.lifetime - 60)
        else:
            max_runtime = str_to_timedelta(max_runtime).seconds
        if max_runtime != float("inf") and max_runtime >= self.lifetime:
            msg = "Allowed task runtime exceeds provider walltime. Tasks might get killed by the scheduler."
            logger.warning(msg)
        self.max_runtime = max_runtime
        if (
            self.executor_type == "workqueue"
            and self.kwargs["min_runtime"] >= self.max_runtime
        ):
            msg = "Minimum task runtime exceeds maximum runtime. WQ might not not start tasks."
            logger.warning(msg)

        # set default WQ resource specs
        self.spec: dict | None = None
        if self.executor_type == "workqueue":
            self.spec = {
                "cores": self.kwargs["cores_per_task"],
                "memory": int(self.kwargs["mem_per_task"] * 1000),  # in MB
                "gpus": self.kwargs["gpus_per_task"],
                "disk": 0,  # not implemented
                "running_time_min": self.kwargs["min_runtime"],
            }
        register_definition(definition=self)

        # TODO: how to handle env variables?
        # disable thread affinity and busy-idling until we can isolate task resources
        default_env_vars = {
            "PYTHONUNBUFFERED": "TRUE",
            "OMP_PROC_BIND": "FALSE",
            "OMP_WAIT_POLICY": "PASSIVE",
            "OMP_DISPLAY_ENV": "VERBOSE",  # verbose OMP log
        }
        if self.executor_type == "threadpool":
            default_env_vars |= {"OMP_NUM_THREADS": f"{self.cores_per_task}"}
        else:
            # WQ sets OMP_NUM_THREADS itself
            pass

        # yaml parsing might un-stringify some keys
        env_vars = {k: str(v).upper() for k, v in (env_vars or {}).items()}
        self.env_vars = default_env_vars | (env_vars or {})

        return

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def lifetime(self) -> float:
        """How long will this manager survive (in seconds)"""
        return self.resources["lifetime"]

    @property
    def use_gpu(self) -> bool:
        if self.executor_type == "threadpool":
            return self.kwargs["use_gpu"]
        return self.kwargs["gpus_per_task"] > 0

    @property
    def cores_per_task(self) -> int:
        if self.executor_type == "workqueue":
            return self.kwargs["cores_per_task"]
        # assumes all threads are working
        cores_per_thread = self.resources["cores"] / self.kwargs["max_threads"]
        return max(int(cores_per_thread), 1)

    @property
    def task_slots(self) -> int:
        if self.executor_type == "threadpool":
            return self.kwargs["max_threads"]

        gpu_slots, memory_slots = float("inf"), float("inf")
        cpu_slots = self.resources["cores"] // self.cores_per_task
        if self.use_gpu:
            gpu_slots = self.resources["gpus"] // self.kwargs["gpus_per_task"]
        if (mem_per_task := self.kwargs["mem_per_task"]) > 0:
            memory_slots = self.resources["memory"] // mem_per_task
        return min(cpu_slots, gpu_slots, memory_slots)

    def wrap_in_timeout(self, command: str) -> str:
        if self.max_runtime == float("inf"):
            return command  # noop

        # send SIGTERM after max_runtime, follow with SIGKILL 30s later
        return f"timeout -k 30s {self.max_runtime}s {command}"

    # def wrap_in_srun(self, command: str) -> str:
    #     # TODO: stub -- this does not work
    #     if self.provider is None:
    #         return command  # noop
    # return f"srun -t 1 -c $CORES {command}"

    def _create_threadpool(self, path: Path) -> ThreadPoolExecutor:
        max_threads = self.kwargs["max_threads"]
        return ThreadPoolExecutor(self.name, max_threads, working_dir=str(path))

    def _create_workqueue(self, path: Path) -> WorkQueueExecutor:
        """See https://cctools.readthedocs.io/en/latest/man_pages/work_queue_worker/#synopsis"""

        # ensure proper scale in # TODO: why is this needed?
        timeout = int(1e6) if self.resources["nodes"] > 1 else 20
        cores = self.resources["cores"]

        worker_options = ["--parent-death", f"--cores={cores}", f"--timeout={timeout}"]
        if (memory := self.resources["memory"]) is not None:
            worker_options.append(f"--memory={memory * 1000}")  # in MB
        if (lifetime := self.lifetime) != float("inf"):
            # allow some margin for WQ startup
            walltime = max(0.95 * lifetime, lifetime - 30)
            worker_options.append(f"-wall-time={walltime}")
        if self.use_gpu:
            gpus = self.resources["gpus"]
            worker_options.append(f"--gpus={gpus}")

        worker_executable = "work_queue_worker"
        if not isinstance(self, ReferenceEvaluation) and self.container:
            # ModelEvaluation / ModelTraining run in container themselves
            # Reference launches tasks in container
            prepend = self.container.launch_command()
            worker_executable = f"{prepend} {worker_executable}"

        executor = WorkQueueExecutor(
            label=self.name,
            working_dir=str(path / self.name),
            provider=self.provider,
            shared_fs=True,
            port=0,  # avoid multiple executors trying to use the same port
            worker_options=" ".join(worker_options),
            worker_executable=worker_executable,
            scaling_cores_per_worker=cores,
        )
        return executor

    def create_executor(self, path: Path) -> ParslExecutor:
        if self.executor_type == "threadpool":
            return self._create_threadpool(path)
        return self._create_workqueue(path)

    def wq_resources(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        executor: str = "workqueue",
        container: Optional[ContainerSpec] = None,
        **kwargs,
    ):
        if executor == "threadpool":
            ensure(container is None, msg="Threadpool not compatible with containers")
            ensure("max_threads" in kwargs, msg="Specify 'max_threads' for parallelism")
            ensure(
                "slurm" not in kwargs,
                msg="Threadpool not compatible with remote execution",
            )
            executor_kwargs = {
                "max_threads": kwargs["max_threads"],
                "use_gpu": kwargs.get("use_gpu", False),
            }
        elif executor == "workqueue":
            executor_kwargs = {
                "cores_per_task": kwargs.get("cores_per_task", 1),
                "gpus_per_task": kwargs.get("gpus_per_task", 0),
                "mem_per_task": kwargs.get("mem_per_task", 0),
            }
            if executor_kwargs["cores_per_task"] == 0:
                raise ConfigurationError("WQ needs at least one core to launch tasks")
            min_runtime = kwargs.get("min_runtime", "00:00:00")
            executor_kwargs["min_runtime"] = str_to_timedelta(min_runtime).seconds
        else:
            raise ConfigurationError("Invalid executor key")

        # search for Parsl ExecutionProvider block, defaulting to "local"
        if "slurm" in kwargs:
            # use SlurmLauncher if multi-node blocks are requested TODO: what does this fix?
            provider, resources = make_slurm_provider(kwargs["slurm"])
            launcher = SlurmLauncher() if resources["nodes"] > 1 else SimpleLauncher()
            provider.launcher = launcher
        else:
            provider, resources = make_local_provider(kwargs.get("local", {}))
        if executor == "threadpool":
            provider = None  # no provider needed

        return cls(
            provider=provider,
            executor_type=executor,
            executor_kwargs=executor_kwargs,
            resources=resources,
            container=container,
            **kwargs,
        )


class ModelEvaluation(ExecutionDefinition):
    def __init__(
        self,
        timeout: float = 10.0,
        max_resource_multiplier: int | None = None,
        allow_oversubscription: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if self.use_gpu and self.kwargs["gpus_per_task"] > 1:
            raise ConfigurationError("No Hamiltonian can do multi-GPU evaluation")

        # i-Pi will kill client connections after no response for timeout seconds
        self.timeout = timeout

        # allow MD tasks to consume more computational resources based on walkers and hamiltonians
        # but never more than available in a single resource block
        if max_resource_multiplier is None:
            max_resource_multiplier = self.task_slots
        elif max_resource_multiplier > self.task_slots:
            logger.warning(
                "Provided 'max_resource_multiplier' exceeds available task slots "
                f"({max_resource_multiplier} -> {self.task_slots}). "
                f"Limiting 'max_resource_multiplier'."
            )
            max_resource_multiplier = self.task_slots
        self.max_resource_multiplier = max_resource_multiplier

        # whether i-Pi clients are allowed to share cores/GPUs
        self.allow_oversubscription = allow_oversubscription

    def server_command(self) -> str:
        command = "psiflow-server"
        return self.wrap_in_timeout(command)

    def get_driver_resources(self, n_walkers: int, n_drivers: int) -> list[dict]:
        """Divide 'expensive' drivers over available resources."""
        n_clients = n_walkers * n_drivers
        m = self.max_resource_multiplier

        if n_drivers > m and not self.allow_oversubscription:
            # the combination of drivers does not fit on available resources
            raise ConfigurationError(
                f"Simulation with {n_drivers} independent drivers not possible. "
                f"Either increase 'max_resource_multiplier' or enable resource oversubscription."
            )
        if n_clients > m and self.allow_oversubscription:
            logger.warning(
                f"Simulation wants to employ {n_clients} clients, "
                f"but can only use {m}x the per-client budget. "
                f"Oversubscribing CPU/GPU resources."
            )
        elif n_clients > m and not self.allow_oversubscription:
            # limit total numer of clients so they do not fight over resources
            n_clients = m

        # TODO: what if (n_clients % n_drivers != 0)
        #  you will have more copies of some drivers and fewer of others..
        # TODO: what if (n_clients % m != 0)
        #  you will have more clients on some GPUs than others

        if not self.use_gpu:
            return [{"device": "cpu"} for _ in range(n_clients)]
        return [{"device": f"cuda:{_ % m}"} for _ in range(n_clients)]

    def wq_resources(self, n_clients: int) -> dict:
        if self.spec is None:
            return {}  # threadpool

        spec = self.spec.copy()
        multi = min(n_clients, self.max_resource_multiplier)
        spec["cores"] *= multi
        spec["gpus"] *= multi
        spec["memory"] *= multi
        return spec


class ModelTraining(ExecutionDefinition):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if not self.use_gpu:
            logger.warning(
                "ModelTraining is configured for CPU operation. Is this what you want?"
            )

        # if self.multigpu:
        #     # TODO: why? Think this might be a multinode thing - which I do not care about
        #     message = (
        #         "the max_training_time keyword does not work "
        #         "in combination with multi-gpu training. Adjust "
        #         "the maximum number of epochs to control the "
        #         "duration of training"
        #     )
        #     assert self.max_runtime is None, message

    def train_command(self, initialize: bool = False):
        command = "psiflow-mace-train"
        return self.wrap_in_timeout(command)

    def wq_resources(self, *args, **kwargs) -> dict:
        if self.spec is None:
            return {}  # threadpool
        return self.spec.copy()


class ReferenceEvaluation(ExecutionDefinition):
    def __init__(
        self,
        reference: ReferenceSpec,
        memory_limit: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reference = reference
        self.memory_limit = memory_limit  # in GB

        if self.use_gpu:
            logger.warning("Reference calculations do not support GPU computation yet.")

    def command(self):
        command = self.reference.launch_command()
        kwargs = {k: getattr(self, k) for k in self.reference.reference_args}
        command = command.format(**kwargs)

        if self.container is not None:
            command = f"{self.container.launch_command()} {command}"
        if (mem := self.memory_limit) is not None:
            # set max RAM usage and disable swap storage - requires systemd-run
            command = f"systemd-run --user --scope -p MemoryMax={mem}G -p MemorySwapMax=0 {command}"

        return self.wrap_in_timeout(command)

    def wq_resources(self, n_cores: int | None) -> dict:
        if self.spec is None:
            return {}  # threadpool

        fraction = 1
        if n_cores is not None:
            fraction = n_cores / self.kwargs["cores_per_task"]
        spec = self.spec.copy()
        spec["cores"] = int(spec["cores"] * fraction)
        spec["memory"] *= fraction
        return spec

    @property
    def name(self) -> str:
        if not hasattr(self, "reference"):
            return super().name  # during init
        return self.reference.name


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
        tmpdir_root: str,
        keep_tmpdirs: bool,
        **kwargs,
    ) -> None:
        self.config = config
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)

        self.definitions = {d.name: d for d in definitions}
        ensure(len(self.definitions) == len(definitions))

        # make sure task tmpdirs can be made
        Path(tmpdir_root).mkdir(parents=True, exist_ok=True)
        self.bash_template = create_bash_template(tmpdir_root, keep_tmpdirs)
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
        assert prefix[-1] == "_"
        assert suffix[0] == "."
        padding = 6
        with self.lock:
            key = (prefix, suffix)
            if key not in self.file_index.keys():
                self.file_index[key] = 0
            assert self.file_index[key] < (16**padding)
            identifier = f"{self.file_index[key]:0{padding}x}"
            self.file_index[key] += 1
            return File(str(self.path / (prefix + identifier + suffix)))

    @classmethod
    def from_config(
        cls,
        parsl_log_level: str,
        psiflow_log_level: str,
        default_threads: int,
        **kwargs,
    ) -> "ExecutionContext":
        path = Path.cwd().resolve() / PSIFLOW_INTERNAL
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        patch_parsl_dirtree()

        # setup logging
        log_file = str(path / PARSL_LOGFILE)
        log_level = getattr(logging, parsl_log_level)
        parsl.set_file_logger(filename=log_file, name="parsl", level=log_level)
        setup_logging(file=path / PSIFLOW_LOGFILE, level=psiflow_log_level)

        # default container for ModelEvaluation and ModelTraining
        base_container = None
        if "container" in kwargs:
            base_container = make_cls(ContainerSpec, **kwargs["container"])

        # create definitions
        model_evaluation = ModelEvaluation.from_config(
            container=base_container, **kwargs["ModelEvaluation"]
        )
        model_training = ModelTraining.from_config(
            container=base_container, **kwargs["ModelTraining"]
        )
        reference_evaluations = []  # reference evaluations are class specific
        for key, reference_cls in REFERENCE_SPECS.items():
            if key in kwargs:
                config = kwargs[key]
                container = None
                if "container" in config:
                    container = make_cls(ContainerSpec, **config.pop("container"))
                reference = make_cls(reference_cls, **config)
                reference_evaluation = ReferenceEvaluation.from_config(
                    reference=reference,
                    container=container,
                    **config,  # make sure the container key is removed
                )
                reference_evaluations.append(reference_evaluation)
        definitions = [model_evaluation, model_training, *reference_evaluations]

        # create main parsl executors
        executors = [d.create_executor(path=path) for d in definitions]
        internal = make_default_executors(default_threads, path, base_container)
        executors.extend(internal)

        config = make_cls(
            Config,
            executors=executors,
            run_dir=str(path),
            initialize_logging=False,
            **kwargs,
        )
        return ExecutionContext(config, definitions, path / CONTEXT_DIR, **kwargs)


class ExecutionContextLoader:
    _context: Optional[ExecutionContext] = None

    @classmethod
    def load(
        cls,
        config: Optional[dict[str, Any]] = None,
    ) -> ExecutionContext:
        if cls._context is not None:
            raise RuntimeError("ExecutionContext has already been loaded")
        if config is not None:
            pass
        elif len(sys.argv) == 1:
            config = {}
        else:
            assert len(sys.argv) <= 2  # only accept a single argument
            path_config = Path(sys.argv[1])
            assert path_config.exists()
            assert path_config.suffix in [".yaml", ".yml"], (
                f"the execution configuration needs to be specified"
                f" as a YAML file, but got {path_config}"
            )
            with open(path_config, "r") as f:
                config = yaml.safe_load(f)

        # set the context so it can be retrieved later
        config = yaml.safe_load(DEFAULT_CONFIG) | config
        cls._context = ExecutionContext.from_config(**config)
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
    # TODO: what does this do?
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


def patch_parsl_dirtree() -> None:
    """By default, Parsl will put Executor logs etc. under numbered directories.
    We do not need this level of nesting, as psiflow_internal is refreshed every run"""
    import parsl.dataflow.dflow

    # replace with noop, which needs to happen after parsl.dataflow.dflow initialises
    parsl.dataflow.dflow.make_rundir = lambda x: x


# TODO: after 3.12, this is no longer needed
#  https://docs.python.org/3/library/typing.html
T = TypeVar("T")


def make_cls(cls: type[T], **kwargs: Any) -> T:
    """Very simple class factory. Use introspection to filter args and kwargs."""
    sign = inspect.signature(cls)
    argument_names = list(sign.parameters.keys())
    arguments = {k: kwargs[k] for k in argument_names if k in kwargs}
    return cls(**arguments)
