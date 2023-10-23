from __future__ import annotations  # necessary for type-guarding class methods

import argparse
import atexit
import importlib
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
# see https://stackoverflow.com/questions/59904631/python-class-constants-in-dataclasses
from typing import Callable, Optional, Type, Union

import parsl
import typeguard
from parsl.config import Config
from parsl.data_provider.files import File
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider
from parsl.providers.base import ExecutionProvider

from psiflow.models import BaseModel
from psiflow.parsl_utils import MyWorkQueueExecutor
from psiflow.reference import BaseReference
from psiflow.utils import set_logger

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
@dataclass(frozen=True, eq=True)  # allows checking for equality
class ExecutionDefinition:
    parsl_provider: ExecutionProvider
    gpu: bool = False
    cores_per_worker: int = 1
    max_walltime: Optional[float] = None

    def __post_init__(self):
        if hasattr(self.parsl_provider, "walltime"):
            walltime_hhmmss = self.parsl_provider.walltime.split(":")
            assert len(walltime_hhmmss) == 3
            walltime = 0
            walltime += 60 * float(walltime_hhmmss[0])
            walltime += float(walltime_hhmmss[1])
            walltime += 1  # whatever seconds are present
            walltime -= 5  # add 5 minutes of slack, e.g. for container downloading
            if self.max_walltime is None:
                object.__setattr__(self, "max_walltime", walltime)  # avoid frozen
            else:  # check whether it doesn't exceeed it
                assert (
                    walltime > self.max_walltime
                ), "{} walltime must be larger than max_walltime".format(
                    type(self.parsl_provider)
                )
        elif self.max_walltime is None:
            object.__setattr__(self, "max_walltime", 1e9)  # avoid frozen

    def generate_parsl_resource_specification(self):
        resource_specification = {}
        resource_specification["cores"] = self.cores_per_worker
        resource_specification["disk"] = 1000
        memory = 2000 * self.cores_per_worker
        resource_specification["memory"] = int(memory)
        if self.max_walltime is not None:
            resource_specification["running_time_min"] = self.max_walltime
        return resource_specification

    def name(self):
        return self.__class__.__name__


@typeguard.typechecked
@dataclass(frozen=True, eq=True)
class Default(ExecutionDefinition):
    pass


@typeguard.typechecked
@dataclass(frozen=True, eq=True)
class ModelEvaluation(ExecutionDefinition):
    simulation_engine: str = "openmm"

    def __post_init__(self) -> None:  # validate config
        super().__post_init__()
        assert self.simulation_engine in ["openmm", "yaff"]


@typeguard.typechecked
@dataclass(frozen=True, eq=True)
class ModelTraining(ExecutionDefinition):
    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.gpu


@typeguard.typechecked
@dataclass(frozen=True, eq=True)
class ReferenceEvaluation(ExecutionDefinition):
    mpi_command: Callable = (
        lambda x: f"mpirun -np {x} -bind-to core -rmk user -launcher fork"
    )
    reference_calculator: Optional[Type[BaseReference]] = None

    def name(self):
        if self.reference_calculator is not None:
            return self.reference_calculator.__name__
        else:
            return super().name()


@typeguard.typechecked
def get_psiflow_config_from_file(
    path_config: Union[Path, str],
    path_internal: Union[Path, str],
) -> tuple[Config, list]:
    if not path_config == "":
        path_config = Path(path_config)
        assert path_config.is_file(), "cannot find psiflow config at {}".format(
            path_config
        )
        # see https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
        spec = importlib.util.spec_from_file_location("module.name", path_config)
        psiflow_config_module = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = psiflow_config_module
        spec.loader.exec_module(psiflow_config_module)
        return psiflow_config_module.get_config(path_internal)
    else:
        if path_internal.exists():
            shutil.rmtree(path_internal)
        path_internal.mkdir()
        default = Default(
            cores_per_worker=4,
            parsl_provider=LocalProvider(),  # unused
        )
        model_evaluation = ModelEvaluation(
            parsl_provider=LocalProvider(),
            cores_per_worker=4,
            gpu=False,
            simulation_engine="openmm",
        )
        model_training = ModelTraining(
            cores_per_worker=4,
            parsl_provider=LocalProvider(),
            gpu=True,
        )
        reference_evaluation = ReferenceEvaluation(
            parsl_provider=LocalProvider(),
            cores_per_worker=4,
            mpi_command=lambda x: f"mpirun -np {x}",
        )
        definitions = [
            default,
            model_evaluation,
            model_training,
            reference_evaluation,
        ]
        executors = [
            ThreadPoolExecutor(
                label="Default",
                max_threads=4,
                working_dir=str(path_internal),
            ),
            ThreadPoolExecutor(
                label="ModelTraining",
                max_threads=4,
                working_dir=str(path_internal),
            ),
            ThreadPoolExecutor(
                label="ModelEvaluation",
                max_threads=4,
                working_dir=str(path_internal),
            ),
            ThreadPoolExecutor(
                label="ReferenceEvaluation",
                max_threads=4,
                working_dir=str(path_internal),
            ),
        ]
        config = Config(
            executors, run_dir=str(path_internal), usage_tracking=True, app_cache=False
        )
        return config, definitions


@typeguard.typechecked
def generate_parsl_config(
    path_internal: Union[Path, str],
    definitions: list[ExecutionDefinition],
    use_work_queue: bool = True,
    wq_timeout: int = 120,  # in seconds
    parsl_app_cache: bool = False,
    parsl_retries: int = 0,
    parsl_max_idletime: int = 30,  # in seconds
    parsl_strategy: str = "simple",
    parsl_initialize_logging: bool = True,
    htex_address: Optional[str] = None,
) -> Config:
    labels = [d.name() for d in definitions]
    assert len(labels) == len(
        set(labels)
    ), "labels must be unique, but found {}".format(labels)
    executors = []
    for definition in definitions:
        if type(definition) is not Default:
            executor = HighThroughputExecutor(
                address=htex_address,
                label=definition.name(),
                working_dir=str(Path(path_internal) / definition.name()),
                cores_per_worker=1,
                provider=definition.parsl_provider,
            )
        else:
            if use_work_queue:
                worker_options = []
                if hasattr(definition.parsl_provider, "cores_per_node"):
                    worker_options.append(
                        "--cores={}".format(definition.parsl_provider.cores_per_node),
                    )  # multiple workers per node
                else:
                    worker_options.append(  # single worker
                        "--cores={}".format(definition.cores_per_worker),
                    )
                definition.generate_parsl_resource_specification()  # populates max_walltime attr
                if (
                    definition.max_walltime is not None
                ):  # convert from minutes to seconds
                    worker_options.append(
                        "--wall-time={}".format(definition.max_walltime * 60)
                    )
                worker_options.append("--timeout={}".format(wq_timeout))
                worker_options.append("--parent-death")
                executor = MyWorkQueueExecutor(
                    label=definition.name(),
                    working_dir=str(Path(path_internal) / definition.name()),
                    provider=definition.parsl_provider,
                    shared_fs=True,
                    autocategory=False,
                    port=0,
                    max_retries=0,
                    coprocess=False,
                    worker_options=" ".join(worker_options),
                )
            else:
                executor = HighThroughputExecutor(
                    address=htex_address,
                    label=definition.name(),
                    working_dir=str(Path(path_internal) / definition.name()),
                    cores_per_worker=definition.cores_per_worker,
                    provider=definition.parsl_provider,
                )
        executors.append(executor)
    config = Config(
        executors=executors,
        run_dir=str(path_internal),
        usage_tracking=True,
        app_cache=False,
        retries=parsl_retries,
        initialize_logging=False,
        strategy=parsl_strategy,
        max_idletime=parsl_max_idletime,
    )
    return config


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
        self.definitions = definitions
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._apps = {}
        self.file_index = {}
        parsl.load(config)

    def __getitem__(self, Class):
        if issubclass(Class, BaseReference):
            for definition in self.definitions:
                if isinstance(definition, ReferenceEvaluation):
                    if definition.reference_calculator == Class:
                        return definition
            for definition in self.definitions:
                if isinstance(definition, ReferenceEvaluation):
                    if definition.reference_calculator is None:
                        return definition
        if issubclass(Class, BaseModel):
            for evaluation in self.definitions:
                if isinstance(evaluation, ModelEvaluation):
                    for training in self.definitions:
                        if isinstance(training, ModelTraining):
                            return evaluation, training
        raise ValueError(
            "no available execution definition for {}".format(Class.__name__)
        )

    def apps(self, Class, app_name: str) -> Callable:
        assert app_name in self._apps[Class].keys()
        return self._apps[Class][app_name]

    def register_app(
        self,
        Class,
        app_name: str,
        app: Callable,
    ) -> None:
        if Class not in self._apps.keys():
            self._apps[Class] = {}
        assert app_name not in self._apps[Class].keys()
        self._apps[Class][app_name] = app

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
        psiflow_log_level: Union[int, str] = "INFO",
        parsl_log_level: Union[int, str] = "INFO",
    ) -> ExecutionContext:
        if cls._context is not None:
            raise RuntimeError("ExecutionContext has already been loaded")
        if path_config is None:  # assume all options are given via command line
            parser = argparse.ArgumentParser()
            parser.add_argument("--psiflow-config", default="", type=str)
            parser.add_argument("--path-internal", default="psiflow_internal", type=str)
            parser.add_argument("--psiflow-log-level", default="INFO", type=str)
            parser.add_argument("--parsl-log-level", default="INFO", type=str)
            args = parser.parse_args()

            path_config = args.psiflow_config
            path_internal = args.path_internal
            psiflow_log_level = args.psiflow_log_level
            parsl_log_level = args.parsl_log_level

        # convert all paths into absolute paths as this is necessary when using
        # WQ executor with shared_fs=True
        if not path_config == "":
            path_internal = Path(path_internal).resolve()
        else:
            path_internal = Path.cwd() / ".psiflow_internal"
        config, definitions = get_psiflow_config_from_file(
            path_config,
            path_internal,
        )
        if not path_internal.is_dir():
            path_internal.mkdir()
        else:
            assert not any(path_internal.iterdir()), "{} should be empty".format(
                str(path_internal)
            )
        set_logger(psiflow_log_level)
        if isinstance(parsl_log_level, str):
            parsl_log_level = getattr(logging, parsl_log_level)
        parsl.set_file_logger(
            str(path_internal / "parsl.log"),
            "parsl",
            parsl_log_level,
            format_string="%(levelname)s - %(name)s - %(message)s",
        )
        path_context = path_internal / "context_dir"
        cls._context = ExecutionContext(config, definitions, path_context)
        atexit.register(parsl.wait_for_current_tasks)
        return cls._context

    @classmethod
    def context(cls):
        if cls._context is None:
            raise RuntimeError("No ExecutionContext is currently loaded")
        return cls._context
