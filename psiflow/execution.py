import logging
import re
import shutil
import sys
import warnings
import subprocess
import textwrap
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

# see https://stackoverflow.com/questions/59904631/python-class-constants-in-dataclasses
from typing import Any, Optional, Union, ClassVar, Protocol, Iterable, Sequence

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


logger = logging.getLogger(__name__)  # logging per module


PSIFLOW_INTERNAL = "psiflow_internal"  # TODO: move configuration files somewhere


@dataclass
class ContainerSpec:
    """Controls container configuration"""

    uri: str
    engine: str = "apptainer"
    addopts: str = " --no-eval -e --no-mount home -W /tmp --writable-tmpfs"
    gpu_flavour: str | None = None  # TODO: add yaml argument

    def __post_init__(self):
        assert self.engine in ("apptainer", "singularity")
        assert len(self.uri) > 0
        assert self.gpu_flavour in ("cuda", "rocm", None)

    def launch_command(self) -> str:
        pwd = Path.cwd().resolve()  # access to data / internal dir
        args = [self.engine, "exec", self.addopts, f"--bind {pwd}"]
        if self.gpu_flavour == "cuda":
            args.append("--nv")
        elif self.gpu_flavour == "rocm":
            args.append("--rocm")
        return " ".join(args)

    @staticmethod
    def from_kwargs(kwargs: dict) -> Optional["ContainerSpec"]:
        if "container_uri" not in kwargs:
            return None
        keys = ("container_uri", "container_engine", "container_addopts")
        args = [kwargs[key] for key in keys if key in kwargs]
        return ContainerSpec(*args)


class ReferenceSpec(Protocol):
    """Defines default options for Reference implementations"""

    name: ClassVar[str]
    reference_args: ClassVar[tuple[str, ...]]  # TODO: update 'cores_per_worker'
    mpi_command: str
    mpi_args: Sequence[str]
    executable: str

    def launch_command(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_kwargs(cls, **kwargs):
        keys = ("mpi_command", "mpi_args", "executable")
        return cls(**{k: kwargs[k] for k in keys if k in kwargs})


@dataclass
class CP2KReferenceSpec(ReferenceSpec):
    name = "CP2K"
    reference_args = ("cores_per_worker",)
    mpi_command: str = "mpirun -np {cores_per_worker}"
    mpi_args: Sequence[str] = (
        "-ENV OMP_NUM_THREADS=1",
        "--bind-to core",
        "--map-by core",
    )
    executable: str = "cp2k.psmp -i cp2k.inp"

    def launch_command(self):
        return " ".join([self.mpi_command, *self.mpi_args, self.executable])


@dataclass
class GPAWReferenceSpec(ReferenceSpec):
    name = "GPAW"
    reference_args = ("cores_per_worker",)
    mpi_command: str = "mpirun -np {cores_per_worker}"
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
    required = ("cores_per_node", "walltime", "gpus_per_node")
    kwargs = defaults | kwargs
    assert all(key in kwargs for key in required)
    provider = SlurmProvider(**kwargs)  # does not configure Launcher
    resources = {
        "nodes": provider.nodes_per_block,
        "cores": provider.cores_per_node,
        "memory": provider.mem_per_node,
        "gpus": provider.gpus_per_node,
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
        self.env_vars = env_vars or {}

        if self.use_gpu:
            msg = ""
            if resources["gpus"] == 0:
                msg = "GPU usage requested but no GPUs available"
            elif container is not None and container.gpu_flavour is None:
                msg = "Provide container 'gpu_flavour' to choose between CUDA and ROCM"
            if msg:
                raise ValueError(msg)

        if self.executor_type == "workqueue":
            # WQ-specific checks TODO: check that WQ kwargs do not exceed resources?
            msg = ""
            if self.kwargs["gpus_per_task"] > resources["gpus"]:
                msg = "GPUs"
            if self.kwargs["cores_per_task"] > resources["cores"]:
                msg = "cores"
            if self.kwargs["mem_per_task"] > (resources["memory"] or float("inf")):
                # TODO: do we need memory=None anywhere? otherwise default to inf?
                msg = "memory"
            if msg:
                msg = f"Apps will request more {msg} than available per Parsl block"
                raise ValueError(msg)

        # how long can individual tasks run (in seconds)
        if max_runtime is None:
            # allow some margin for task cleanup  TODO: pretty random
            max_runtime = max(0.9 * self.lifetime, self.lifetime - 60)
        else:
            max_runtime = str_to_timedelta(max_runtime).seconds
        if max_runtime != float("inf") and max_runtime >= self.lifetime:
            msg = "Allowed task runtime exceeds provider walltime. Tasks might get killed by the scheduler."
            warnings.warn(msg)
        self.max_runtime = max_runtime

        # set default WQ resource specs  TODO: type_hint
        self.spec = None
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
        # TODO: check between min_runtime and max_runtime?

        pass

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
        return int(self.resources["cores"] / self.kwargs["max_threads"])

    def wrap_in_timeout(self, command: str) -> str:
        if self.max_runtime == float("inf"):
            return command  # noop

        # send SIGTERM after max_runtime, follow with SIGKILL 30s later
        return f"timeout -k 30s {self.max_runtime}s {command}"

    def _create_threadpool(self, path: Path) -> ThreadPoolExecutor:
        max_threads = self.kwargs["max_threads"]
        return ThreadPoolExecutor(self.name, max_threads, working_dir=str(path))

    def _create_workqueue(self, path: Path) -> WorkQueueExecutor:
        """See https://cctools.readthedocs.io/en/latest/man_pages/work_queue_worker/#synopsis"""

        # ensure proper scale in # TODO: why is this needed?
        timeout = int(1e6) if self.resources["nodes"] > 1 else 20
        cores = self.resources["cores"]

        worker_options = [
            "--parent-death",
            f"--cores={cores}",
            f"--timeout={timeout}",
        ]
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
            # Reference instances launch tasks in container
            prepend = self.container.launch_command()
            worker_executable = f"{prepend} {worker_executable}"

        # TODO: why the custom WQ? -- does not seem necessary (anymore)
        # executor = MyWorkQueueExecutor(
        executor = WorkQueueExecutor(
            label=self.name,
            working_dir=str(path / self.name),
            provider=self.provider,
            shared_fs=True,
            # autocategory=False,
            # port=0,
            # max_retries=1,
            # coprocess=False,
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
        executor: str,  # TODO: no default value?
        container: Optional[ContainerSpec],
        **kwargs,
    ):
        if executor == "threadpool":
            assert container is None, "Threadpool not compatible with containers"
            assert (
                "slurm" not in kwargs
            ), "Threadpool not compatible with remote execution"
            assert "max_threads" in kwargs, "Specify 'max_threads' for parallelism"
            executor_kwargs = {
                "max_threads": kwargs["max_threads"],
                "use_gpu": kwargs.get("use_gpu", False),
            }
        elif executor == "workqueue":
            executor_kwargs = {
                "cores_per_task": kwargs.get("cores_per_task", 0),
                "gpus_per_task": kwargs.get("gpus_per_task", 0),
                "mem_per_task": kwargs.get("mem_per_task", 0),
            }
            assert any(v != 0 for v in executor_kwargs.values())
            min_runtime = kwargs.get("min_runtime", "00:00:00")
            executor_kwargs["min_runtime"] = str_to_timedelta(min_runtime).seconds
        else:
            raise ValueError("Key 'executor' must be 'threadpool' or 'workqueue'")

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
        timeout: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.timeout = timeout  # i-Pi will kill client connections after no response for timeout seconds

        if self.executor_type == "threadpool":
            # disable thread affinity and busy-idling
            env_vars = {
                "OMP_PROC_BIND": "FALSE",
                "OMP_WAIT_POLICY": "PASSIVE",
                "OMP_NUM_THREADS": f"{self.cores_per_task}",
                # "OMP_DISPLAY_ENV": "VERBOSE",  # verbose OMP log
            }
        else:
            assert False, "IMPLEMENT THIS"
        self.env_vars = env_vars | self.env_vars

    def server_command(self) -> str:
        command = "psiflow-server"
        return self.wrap_in_timeout(command)

    def get_driver_devices(self, nwalkers: int) -> list[dict]:
        # assumes driver is GPU capable
        # TODO: what if only 1 gpu is available? Redo this
        # nclients = min(nwalkers, self.max_workers)
        nclients = min(nwalkers, 2)
        if self.use_gpu:
            return [{"device": f"cuda:{i}"} for i in range(nclients)]
        else:
            return [{"device": "cpu"} for _ in range(nclients)]

    def wq_resources(self, nwalkers: int) -> dict:
        if self.spec is None:
            return {}  # threadpool
        # TODO: reimplement this
        return self.spec

    # def wq_resources(self, nwalkers):
    #     if self.use_threadpool:
    #         return {}
    #     nclients = min(nwalkers, self.max_workers)
    #     resource_specification = {}
    #     resource_specification["cores"] = nclients * self.cores_per_worker
    #     resource_specification["disk"] = 1000  # some random nontrivial amount?
    #     memory = 2000 * self.cores_per_worker  # similarly rather random
    #     resource_specification["memory"] = int(memory)
    #     resource_specification["running_time_min"] = self.max_simulation_time
    #     if self.gpu:
    #         resource_specification["gpus"] = nclients
    #     return resource_specification


class ModelTraining(ExecutionDefinition):
    def __init__(
        self,
        multigpu: bool = False,  # TODO: how to handle this?
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.multigpu = multigpu
        if self.multigpu:
            # TODO: why? Think this might be a multinode thing - which I do not care about
            message = (
                "the max_training_time keyword does not work "
                "in combination with multi-gpu training. Adjust "
                "the maximum number of epochs to control the "
                "duration of training"
            )
            assert self.max_runtime is None, message

        if not self.use_gpu:
            warnings.warn(
                "ModelTraining is configured for CPU operation. Is this what you want?"
            )

        # default_env_vars = {
        #     "OMP_NUM_THREADS": str(self.cores_per_worker),
        #     "KMP_AFFINITY": "granularity=fine,compact,1,0",
        #     "KMP_BLOCKTIME": "1",
        #     "OMP_PROC_BIND": "spread",  # different from Model Eval
        #     "PYTHONUNBUFFERED": "TRUE",
        # }
        # if env_vars is None:
        #     env_vars = default_env_vars
        # else:
        #     default_env_vars.update(env_vars)
        #     env_vars = default_env_vars

    def train_command(self, initialize: bool = False):
        command = "psiflow-mace-train"
        return self.wrap_in_timeout(command)

    def wq_resources(self, *args, **kwargs) -> dict:
        if self.spec is None:
            return {}  # threadpool
        # TODO: reimplement this
        return self.spec

    # def wq_resources(self):
    #     if self.use_threadpool:
    #         return {}
    #     resource_specification = {}
    #
    #     if self.multigpu:
    #         nworkers = int(self.cores_available / self.cores_per_worker)
    #     else:
    #         nworkers = 1
    #
    #     resource_specification["gpus"] = nworkers  # one per GPU
    #     resource_specification["cores"] = self.cores_available
    #     resource_specification["disk"] = (
    #         1000 * nworkers
    #     )  # some random nontrivial amount?
    #     memory = 1000 * self.cores_available  # similarly rather random
    #     resource_specification["memory"] = int(memory)
    #     resource_specification["running_time_min"] = self.max_training_time
    #     return resource_specification


class ReferenceEvaluation(ExecutionDefinition):
    def __init__(
        self,
        reference: "ReferenceSpec",
        memory_limit: Optional[str] = None,  # TODO: how does this work?
        **kwargs,
    ) -> None:
        # TODO: how to know which code?
        # before super().__init__ because 'name' attribute needed
        self.reference = reference
        super().__init__(**kwargs)
        self.memory_limit = memory_limit

    def command(self):
        # TODO: this does not work probably
        launch_command = self.spec.launch_command()
        kwargs = {k: getattr(self, k) for k in self.spec.reference_args}
        launch_command = launch_command.format(**kwargs)

        if self.container is not None:
            launch_command = f"{self.container.launch_command()} {launch_command}"

        launch_command = self.wrap_in_timeout(launch_command)

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

    def wq_resources(self, *args, **kwargs) -> dict:
        if self.spec is None:
            return {}  # threadpool
        return self.spec

    @property
    def name(self) -> str:
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
        assert len(self.definitions) == len(definitions)

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
        default_threads: int,
        **kwargs,
    ) -> "ExecutionContext":
        path = Path.cwd().resolve() / PSIFLOW_INTERNAL
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        patch_parsl_dirtree()

        log_file = str(path / "parsl.log")
        log_level = getattr(logging, parsl_log_level)
        parsl.set_file_logger(filename=log_file, name="parsl", level=log_level)

        # create definitions
        base_container = ContainerSpec.from_kwargs(kwargs)
        model_evaluation = ModelEvaluation.from_config(
            container=base_container, **kwargs["ModelEvaluation"]
        )
        model_training = ModelTraining.from_config(
            container=base_container, **kwargs["ModelTraining"]
        )

        reference_evaluations = []  # reference evaluations might be class specific
        for key in list(kwargs.keys()):
            if key[:4] in REFERENCE_SPECS:  # allow for e.g., CP2K_small
                config = kwargs[key]
                reference_evaluation = ReferenceEvaluation.from_config(
                    reference=REFERENCE_SPECS[key[:4]].from_kwargs(**config),
                    container=ContainerSpec.from_kwargs(kwargs | config),
                    **config,
                )
                reference_evaluations.append(reference_evaluation)
        definitions = [model_evaluation, model_training, *reference_evaluations]

        # create main parsl executors
        executors = [d.create_executor(path=path) for d in definitions]
        internal = make_default_executors(default_threads, path, base_container)
        executors.extend(internal)

        config = Config(
            executors=executors, run_dir=str(path), initialize_logging=False
        )
        return ExecutionContext(config, definitions, path / "context_dir", **kwargs)


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


# class MyWorkQueueExecutor(WorkQueueExecutor):
#     # TODO: what does this do?
#     def _get_launch_command(self, block_id):
#         return self.worker_command


# TODO: move everything below to appropriate files

# TODO: attempt at managing priority through global state
WQ_RESOURCES_REGISTRY = {}


def register_definition(definition: ExecutionDefinition) -> None:
    """"""
    if (spec := definition.spec) is None:
        return  # threadpool does not have priority

    WQ_RESOURCES_REGISTRY[definition.name] = spec
    spec["priority"] = SetWQPriority.default


class SetWQPriority:
    """Manage the WQ priority tag as context manager"""

    # TODO: this probably does not work in a nested way
    # TODO: log to parsl.log?
    default = 0

    def __init__(self, value: int, verbose: bool = False) -> None:
        self.value = value
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(f"SetWQPriority setting priority:\t{self.value}")
        for n, spec in WQ_RESOURCES_REGISTRY.items():
            spec["priority"] = self.value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print(f"SetWQPriority unsetting {self.value}")
        for n, spec in WQ_RESOURCES_REGISTRY.items():
            spec["priority"] = SetWQPriority.default


# This is the default psiflow config which is always passed into the ExecutionContext
# TODO: find a place for this
DEFAULT_CONFIG = """
parsl_log_level: WARNING
usage_tracking: 3
default_threads: 4
max_idletime: 20
tmpdir_root: /tmp
keep_tmpdirs: false
gpu_flavour: nvidia

ModelEvaluation:
  executor: threadpool
  max_threads: 2

ModelTraining:
  executor: threadpool
  max_threads: 2
"""


def patch_parsl_dirtree() -> None:
    """By default, Parsl will put Executor logs etc. under numbered directories.
    We do not need this level of nesting, as psiflow_internal is refreshed every run"""
    import parsl.dataflow.dflow

    # replace with noop, which needs to happen after parsl.dataflow.dflow initialises
    parsl.dataflow.dflow.make_rundir = lambda x: x


# TODO: arguments that need documenting: retries, strategy?, timeout, garbage_collect (Config)


def create_bash_template(tmpdir_root: str, keep_tmpdirs: bool) -> str:
    """Create general wrapper for all bash apps. The exitcode ensures that every app completes successfully."""
    template = f"""
    # Create and move into new tmpdir for app execution
    tmpdir=$(mktemp -d -p {tmpdir_root} "psiflow-tmp.XXXXXXXXXX")
    cd $tmpdir; echo "tmpdir: $PWD"
    export {{env}}
    printenv

    # Actual app definition goes here
    {{commands}}

    # Cleanup
    {'cd ../.. && rm -r $tmpdir' if not keep_tmpdirs else ''}
    exit 0
    """
    return textwrap.dedent(template)


def format_env_vars(env_vars: dict) -> str:
    return " ".join([f"{k}={v}" for k, v in env_vars.items()])


def str_to_timedelta(s: str) -> timedelta:
    t = datetime.strptime(s, "%H:%M:%S")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)