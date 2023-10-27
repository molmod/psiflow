from __future__ import annotations  # necessary for type-guarding class methods

import atexit
import copy
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# see https://stackoverflow.com/questions/59904631/python-class-constants-in-dataclasses
from typing import Any, Callable, Optional, Type, Union
from warnings import warn

import parsl
import typeguard
import yaml
from parsl.config import Config
from parsl.data_provider.files import File
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers.launchers import SimpleLauncher
from parsl.providers import *  # noqa: F403
from parsl.providers.base import ExecutionProvider

from psiflow.models import BaseModel
from psiflow.parsl_utils import ContainerizedLauncher
from psiflow.reference import BaseReference
from psiflow.utils import resolve_and_check, set_logger

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
@dataclass
class ExecutionDefinition:
    parsl_provider: ExecutionProvider
    gpu: bool = False
    cores_per_worker: int = 1
    max_walltime: Optional[float] = None
    use_threadpool: bool = False

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
                self.max_walltime = walltime
            else:  # check whether it doesn't exceeed it
                assert (
                    walltime > self.max_walltime
                ), "{} walltime must be larger than max_walltime".format(
                    type(self.parsl_provider)
                )
        elif self.max_walltime is None:
            self.max_walltime = 1e9

    def generate_parsl_resource_specification(self):
        warn("use of the WQ executor is deprecated!")
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
@dataclass
class ModelEvaluation(ExecutionDefinition):
    simulation_engine: str = "openmm"

    def __post_init__(self) -> None:  # validate config
        super().__post_init__()
        assert self.simulation_engine in ["openmm", "yaff"]


@typeguard.typechecked
@dataclass
class ModelTraining(ExecutionDefinition):
    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.gpu


@typeguard.typechecked
@dataclass
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

    @staticmethod
    def parse_config(yaml_dict: dict):
        definitions = []
        for name in ["ModelEvaluation", "ModelTraining", "ReferenceEvaluation"]:
            if name in yaml_dict:
                _dict = yaml_dict.pop(name)
                if type(_dict) is not dict:
                    raise NotImplementedError(
                        "multiple execution definitions per "
                        "category are not yet supported"
                    )
            else:
                _dict = {}

            # set necessary defaults (somewhat duplicate w.r.t. execution definitions)
            defaults = {
                "gpu": False,
                "use_threadpool": False,
                "cores_per_worker": 1,
                "max_walltime": None,
            }
            _dict = dict(defaults, **_dict)

            # if mpi_command is in there, replace it with a lambda
            if "mpi_command" in _dict:
                s = _dict["mpi_command"]
                _dict["mpi_command"] = lambda x, s=s: s.format(x)

            # check if container is requested
            container_kwargs = None
            if "apptainer" in _dict:
                container_kwargs = _dict.pop("apptainer")
                container_kwargs["apptainer_or_singularity"] = "apptainer"
            elif "singularity" in _dict:
                container_kwargs = _dict.pop("singularity")
                container_kwargs["apptainer_or_singularity"] = "singularity"
            if container_kwargs is not None:
                assert not _dict["use_threadpool"]  # not possible with container
                container_kwargs["enable_gpu"] = _dict["gpu"]
                launcher = ContainerizedLauncher(**container_kwargs)
            else:
                launcher = SimpleLauncher()

            # initialize provider
            provider_dict = None
            for key in _dict:
                if "Provider" in key:
                    assert provider_dict is None
                    provider_dict = _dict[key]
            if provider_dict is not None:
                provider_cls = getattr(sys.modules[__name__], key)
                provider = provider_cls(launcher=launcher, **_dict.pop(key))
            else:
                provider = LocalProvider(launcher=launcher)  # noqa: F405

            # initialize definition
            definition_cls = getattr(sys.modules[__name__], name)
            definitions.append(definition_cls(parsl_provider=provider, **_dict))

        # define default values
        defaults = {
            "psiflow_internal": str(Path.cwd().resolve() / "psiflow_internal"),
            "parsl_log_level": "INFO",
            "psiflow_log_level": "INFO",
            "usage_tracking": True,
            "retries": 1,
            "strategy": "htex_auto_scale",
            "max_idletime": 20,
            "htex_address": None,
            "default_threads": 4,
        }
        forced = {
            "initialize_logging": False,  # manual; to move parsl.log one level up
            "app_cache": False,  # disabled; has introduced bugs in the past
        }
        psiflow_config = dict(defaults, **yaml_dict)
        psiflow_config.update(forced)
        return psiflow_config, definitions

    @classmethod
    def load(
        cls,
        psiflow_config: Optional[dict[str, Any]] = None,
        definitions: Optional[list[ExecutionDefinition]] = None,
    ) -> ExecutionContext:
        if cls._context is not None:
            raise RuntimeError("ExecutionContext has already been loaded")
        if psiflow_config is None:  # assume yaml is passed as argument
            if len(sys.argv) == 1:  # no config passed, use default:
                yaml_dict_str = """
---
ModelEvaluation:
  max_walltime: 1000000
  simulation_engine: 'openmm'
  gpu: false
  use_threadpool: true
ModelTraining:
  max_walltime: 1000000
  gpu: true
  use_threadpool: true
ReferenceEvaluation:
  max_walltime: 1000000
  mpi_command: 'mpirun -np {}'
  use_threadpool: true
...
                """
                yaml_dict = yaml.safe_load(yaml_dict_str)
            else:
                assert len(sys.argv) == 2
                path_config = resolve_and_check(Path(sys.argv[1]))
                assert path_config.exists()
                assert path_config.suffix in [".yaml", ".yml"], (
                    "the execution configuration needs to be specified"
                    " as a YAML file, but got {}".format(path_config)
                )
                with open(path_config, "r") as f:
                    yaml_dict = yaml.safe_load(f)
            psiflow_config, definitions = cls.parse_yaml(yaml_dict)
        psiflow_config = copy.deepcopy(psiflow_config)  # modified in place
        path = resolve_and_check(Path(psiflow_config.pop("psiflow_internal")))
        assert not any(path.iterdir()), "internal directory {} should be empty".format(
            path
        )
        set_logger(psiflow_config.pop("psiflow_log_level"))
        parsl.set_file_logger(
            str(path / "parsl.log"),
            "parsl",
            getattr(logging, psiflow_config.pop("parsl_log_level")),
            format_string="%(levelname)s - %(name)s - %(message)s",
        )

        # create parsl executors and config
        executors = []
        for definition in definitions:
            if not definition.use_threadpool:
                executor = HighThroughputExecutor(
                    address=psiflow_config["htex_address"],
                    label=definition.name(),
                    working_dir=str(path / definition.name()),
                    cores_per_worker=definition.cores_per_worker,
                    provider=definition.parsl_provider,
                )
            else:
                executor = ThreadPoolExecutor(
                    max_threads=definition.cores_per_worker,
                    working_dir=str(path),
                    label=definition.name(),
                )
            executors.append(executor)
        executors.append(
            ThreadPoolExecutor(
                max_threads=psiflow_config.pop("default_threads"),
                working_dir=str(path),
                label="Default",
            )
        )

        # remove additional kwargs
        psiflow_config.pop("htex_address")
        config = Config(
            executors=executors,
            run_dir=str(path),
            **psiflow_config,
        )
        path_context = path / "context_dir"
        cls._context = ExecutionContext(config, definitions, path_context)
        atexit.register(parsl.wait_for_current_tasks)
        return cls._context

    @classmethod
    def context(cls):
        if cls._context is None:
            raise RuntimeError("No ExecutionContext is currently loaded")
        return cls._context
