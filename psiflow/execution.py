import logging
import re
import shutil
import sys
import warnings
import subprocess
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

# see https://stackoverflow.com/questions/59904631/python-class-constants-in-dataclasses
from typing import Any, Optional, Union, ClassVar, Protocol, Iterable

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

import psiflow

logger = logging.getLogger(__name__)  # logging per module


PSIFLOW_INTERNAL = "psiflow_internal"
DEFAULT_CONFIG = {  # TODO: remove
    "ModelEvaluation": {"gpu": False, "use_threadpool": True},
    "ModelTraining": {"gpu": True, "use_threadpool": True},
}


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
    reference_args: ClassVar[tuple[str, ...]]
    mpi_command: str
    mpi_args: Iterable[str]
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
    mpi_args: tuple[str, ...] = (
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
    mpi_args: tuple[str, ...] = (
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
    mpi_args: tuple[str, ...] = (
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


def str_to_timedelta(s: str) -> timedelta:
    # TODO: move to utils
    t = datetime.strptime(s, "%H:%M:%S")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)


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
        "cores": kwargs.get("cores", psutil.cpu_count()),
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


class ExecutionDefinition:
    # TODO: do not like defining some kwargs in class method and other kwargs in init...
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
                msg = "Provide 'gpu_flavour' to choose between CUDA and ROCM"
            if msg:
                raise ValueError(msg)

        # how long can individual tasks run (in seconds)
        if max_runtime is None:
            # allow some margin for task cleanup  TODO: pretty random
            max_runtime = max(0.9 * self.lifetime, self.lifetime - 60)
        else:
            max_runtime = str_to_timedelta(max_runtime).seconds
        if max_runtime != float("inf") and max_runtime >= self.lifetime:
            warnings.warn(
                "Allowed task runtime exceeds provider walltime. Tasks might get killed by the scheduler."
            )
        self.max_runtime = max_runtime

        # TODO: check that WQ kwargs do not exceed resources?
        # TODO: how to handle env variables?
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
        return self.kwargs.get("use_gpu") or self.kwargs.get("gpus_per_task") > 0

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

        # TODO: why the custom WQ?
        executor = MyWorkQueueExecutor(
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
        if self.executor_type == "threadpool":
            return {}

        # TODO: why recreate every call?
        # TODO: priority
        spec = {
            "cores": self.kwargs["cores_per_task"],
            "memory": int(self.kwargs["mem_per_task"] * 1000),  # in MB
            "gpus": self.kwargs["gpus_per_task"],
            "disk": 0,  # not implemented
            "running_time_min": self.kwargs["min_runtime"],
        }
        return self._modify_wq_resources(spec, *args, **kwargs)

    def _modify_wq_resources(self, spec: dict, *args, **kwargs) -> dict:
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        executor: str = "workqueue",
        container: Optional[ContainerSpec] = None,
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
        timeout: float = 5,  # TODO: units?
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timeout = timeout

        # TODO: temporary
        self.cores_per_worker = self.kwargs.get("cores_per_task", 1)
        self.gpu = False

        # TODO: what with env vars?
        # default_env_vars = {
        #     "OMP_NUM_THREADS": str(self.cores_per_worker),
        #     "KMP_AFFINITY": "granularity=fine,compact,1,0",
        #     "KMP_BLOCKTIME": "1",
        #     "OMP_PROC_BIND": "false",
        #     "PYTHONUNBUFFERED": "TRUE",
        # }

    def server_command(self) -> str:
        command = "psiflow-server"
        return self.wrap_in_timeout(command)

    def get_driver_devices(self, nwalkers: int) -> list[dict]:
        # assumes driver is GPU capable
        # TODO: what if only 1 gpu is available? Redo this
        # nclients = min(nwalkers, self.max_workers)
        nclients = min(nwalkers, 2)
        if self.gpu:
            return [{"device": f"cuda:{i}"} for i in range(nclients)]
        else:
            return [{"device": "cpu"} for _ in range(nclients)]

    def _modify_wq_resources(self, spec: dict, *args, **kwargs) -> dict:
        pass

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

    def _modify_wq_resources(self, spec: dict, *args, **kwargs) -> dict:
        pass

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
        spec: "ReferenceSpec",
        memory_limit: Optional[str] = None,  # TODO: how does this work?
        **kwargs,
    ) -> None:
        # TODO: how to know which code?
        super().__init__(**kwargs)
        self.spec = spec
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

    def _modify_wq_resources(self, spec: dict, *args, **kwargs) -> dict:
        return spec

    @property
    def name(self) -> str:
        return self.spec.name


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
        parsl_log_level: str = "WARNING",
        usage_tracking: int = 3,
        retries: int = 2,
        strategy: str = "simple",
        max_idletime: float = 20,
        internal_tasks_max_threads: int = 10,
        default_threads: int = 4,
        # htex_address: str = "127.0.0.1",
        zip_staging: Optional[bool] = None,
        make_symlinks: bool = False,
        **kwargs,
    ) -> "ExecutionContext":
        path = Path.cwd().resolve() / PSIFLOW_INTERNAL
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

        log_file = str(path / "parsl.log")
        log_level = getattr(logging, parsl_log_level)
        parsl.set_file_logger(filename=log_file, name="parsl", level=log_level)

        # create definitions
        base_container = ContainerSpec.from_kwargs(kwargs)
        model_evaluation = ModelEvaluation.from_config(
            container=base_container,
            **kwargs.pop("ModelEvaluation", {}),
        )
        model_training = ModelTraining.from_config(
            container=base_container,
            **kwargs.pop("ModelTraining", {"gpu": True}),  # avoid triggering assertion
        )

        # TODO: remove this and check below
        model_evaluation.wq_resources(0)
        model_evaluation.server_command()
        model_training.wq_resources()

        reference_evaluations = []  # reference evaluations might be class specific
        for key in list(kwargs.keys()):
            if key[:4] in REFERENCE_SPECS:  # allow for e.g., CP2K_small
                config = kwargs.pop(key)
                reference_evaluation = ReferenceEvaluation.from_config(
                    # spec=init_spec(REFERENCE_SPECS[key[:4]], config),
                    spec=REFERENCE_SPECS[key[:4]].from_kwargs(**config),
                    container=ContainerSpec.from_kwargs(kwargs | config),
                    **config,
                )
                reference_evaluations.append(reference_evaluation)
        definitions = [model_evaluation, model_training, *reference_evaluations]

        # create main parsl executors
        executors = [d.create_executor(path=path) for d in definitions]

        # create default executors
        # TODO: extract this into function
        if base_container is not None:
            launcher = WrappedLauncher(prepend=base_container.launch_command())
        else:
            launcher = SimpleLauncher()
        htex = HighThroughputExecutor(
            label="default_htex",
            # address=htex_address,
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
            # app_cache=False,
            usage_tracking=usage_tracking,
            retries=retries,
            strategy=strategy,
            max_idletime=max_idletime,
            internal_tasks_max_threads=internal_tasks_max_threads,
            # std_autopath=std_autopath,
        )
        context = ExecutionContext(config, definitions, path / "context_dir")

        # if make_symlinks:
        #     src, dest = Path.cwd() / "psiflow_log", path / "parsl.log"
        #     _create_symlink(src, dest)
        #     src, dest = (
        #         Path.cwd() / "psiflow_submit_scripts",
        #         path / "000" / "submit_scripts",
        #     )
        #     _create_symlink(src, dest, is_dir=True)
        #     src, dest = Path.cwd() / "psiflow_task_logs", path / "000" / "task_logs"
        #     _create_symlink(src, dest, is_dir=True)

        return context


class ExecutionContextLoader:
    _context: Optional[ExecutionContext] = None

    @classmethod
    def load(
        cls,
        config: Optional[dict[str, Any]] = None,
    ) -> ExecutionContext:
        if cls._context is not None:
            raise RuntimeError("ExecutionContext has already been loaded")
        if config is None:
            if len(sys.argv) == 1:  # no yaml config passed, use threadpools:
                config = DEFAULT_CONFIG
            else:
                assert len(sys.argv) == 2
                path_config = psiflow.resolve_and_check(Path(sys.argv[1]))
                assert path_config.exists()
                assert path_config.suffix in [".yaml", ".yml"], (
                    f"the execution configuration needs to be specified"
                    f" as a YAML file, but got {path_config}"
                )
                with open(path_config, "r") as f:
                    config = yaml.safe_load(f)
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


class MyWorkQueueExecutor(WorkQueueExecutor):
    # TODO: what does this do?
    def _get_launch_command(self, block_id):
        return self.worker_command


# def _create_symlink(src: Path, dest: Path, is_dir: bool = False) -> None:
#     """Create or replace symbolic link"""
#     if src.is_symlink():
#         src.unlink()
#     if is_dir:
#         dest.mkdir(parents=True, exist_ok=True)
#     else:
#         dest.touch(exist_ok=True)
#     src.symlink_to(dest, target_is_directory=is_dir)
