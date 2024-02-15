from __future__ import annotations  # necessary for type-guarding class methods

import copy
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# see https://stackoverflow.com/questions/59904631/python-class-constants-in-dataclasses
from typing import Any, Callable, Optional, Type, Union

import parsl
import psutil
import typeguard
import yaml
from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.data_provider.files import File
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers.launchers import SimpleLauncher, SrunLauncher
from parsl.providers import *  # noqa: F403
from parsl.providers.base import ExecutionProvider

from psiflow.models import BaseModel
from psiflow.parsl_utils import ContainerizedLauncher, ContainerizedSrunLauncher
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
    cpu_affinity: str = "block"

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
        resource_specification = {}
        resource_specification["cores"] = self.cores_per_worker
        # add random disk and mem usage because this is somehow required
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

        container_dict = yaml_dict.get("container", None)
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
                "cpu_affinity": "block",
            }
            _dict = dict(defaults, **_dict)

            # if mpi_command is in there, replace it with a lambda
            if "mpi_command" in _dict:
                s = _dict["mpi_command"]
                _dict["mpi_command"] = lambda x, s=s: s.format(x)

            # set up containerized launcher if necessary
            if ("container" not in _dict and container_dict is None) or _dict[
                "use_threadpool"
            ]:
                launcher = SimpleLauncher()
                _container_dict = None
            else:
                _container_dict = yaml_dict.get("container", container_dict)
                assert _container_dict is not None
                launcher = ContainerizedLauncher(
                    **_container_dict,
                    enable_gpu=_dict["gpu"],
                )

            # initialize provider
            provider_keys = list(filter(lambda k: "Provider" in k, _dict.keys()))
            if len(provider_keys) == 0:
                provider = LocalProvider(launcher=launcher)  # noqa: F405
            elif len(provider_keys) == 1:
                provider_dict = _dict.pop(provider_keys[0])

                # if provider requests multiple nodes, switch to (containerized) SrunLauncher
                if provider_dict.get("nodes_per_block", 1) > 1:
                    assert (
                        provider_keys[0] == "SlurmProvider"
                    ), "multi-node blocks only supported for SLURM"
                    if _container_dict is not None:
                        launcher = ContainerizedSrunLauncher(
                            **_container_dict, enable_gpu=_dict["gpu"]
                        )
                    else:
                        launcher = SrunLauncher()
                provider_cls = getattr(sys.modules[__name__], provider_keys[0])
                provider = provider_cls(launcher=launcher, **provider_dict)
            else:
                raise ValueError("Can only have one provider per executor")

            # initialize definition
            definition_cls = getattr(sys.modules[__name__], name)
            definitions.append(definition_cls(parsl_provider=provider, **_dict))

        # define default values
        defaults = {
            "psiflow_internal": str(Path.cwd().resolve() / "psiflow_internal"),
            "parsl_log_level": "INFO",
            "psiflow_log_level": "INFO",
            "usage_tracking": True,
            "retries": 2,
            "strategy": "simple",
            "max_idletime": 20,
            "default_threads": 1,
            "mode": "htex",
            "htex_address": address_by_hostname(),
            "container": None,
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
                path_internal = Path.cwd() / ".psiflow_internal"
                if path_internal.exists():
                    shutil.rmtree(path_internal)
                yaml_dict["psiflow_internal"] = path_internal
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
            psiflow_config, definitions = cls.parse_config(yaml_dict)
        psiflow_config = copy.deepcopy(psiflow_config)  # modified in place
        path = resolve_and_check(Path(psiflow_config.pop("psiflow_internal")))
        if path.exists():
            assert not any(
                path.iterdir()
            ), "internal directory {} should be empty".format(path)
        path.mkdir(parents=True, exist_ok=True)
        parsl.set_file_logger(
            str(path / "parsl.log"),
            "parsl",
            getattr(logging, psiflow_config.pop("parsl_log_level")),
        )
        set_logger(psiflow_config.pop("psiflow_log_level"))

        # create main parsl executors
        executors = []
        mode = psiflow_config.pop("mode")
        htex_address = psiflow_config.pop("htex_address")
        for definition in definitions:
            if definition.use_threadpool:
                executor = ThreadPoolExecutor(
                    max_threads=definition.cores_per_worker,
                    working_dir=str(path),
                    label=definition.name(),
                )
            elif mode == "htex":
                if type(definition.parsl_provider) is LocalProvider:  # noqa: F405
                    cores_available = psutil.cpu_count(logical=False)
                    max_workers = max(
                        1, math.floor(cores_available / definition.cores_per_worker)
                    )
                else:
                    max_workers = float("inf")
                if definition.cores_per_worker == 1:  # anticipate parsl assertion
                    definition.cpu_affinity = "none"
                    logger.info(
                        'setting cpu_affinity of definition "{}" to none '
                        "because cores_per_worker=1".format(definition.name())
                    )
                executor = HighThroughputExecutor(
                    address=htex_address,
                    label=definition.name(),
                    working_dir=str(path / definition.name()),
                    cores_per_worker=definition.cores_per_worker,
                    max_workers=max_workers,
                    provider=definition.parsl_provider,
                    cpu_affinity=definition.cpu_affinity,
                )
            else:
                raise ValueError("Unknown mode {}".format(mode))
            executors.append(executor)

        # create default executors
        container_dict = psiflow_config.pop("container", None)
        if container_dict is not None:
            launcher = ContainerizedLauncher(**container_dict)
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
            max_threads=psiflow_config.pop("default_threads"),
            working_dir=str(path),
        )
        executors.append(threadpool)

        # remove additional kwargs
        config = Config(
            executors=executors,
            run_dir=str(path),
            **psiflow_config,
        )
        path_context = path / "context_dir"
        cls._context = ExecutionContext(config, definitions, path_context)
        return cls._context

    @classmethod
    def context(cls):
        if cls._context is None:
            raise RuntimeError("No ExecutionContext is currently loaded")
        return cls._context

    @classmethod
    def wait(cls):
        parsl.wait_for_current_tasks()


def load_from_yaml(path: Union[str, Path]) -> ExecutionContext:
    assert ExecutionContextLoader._context is None  # no previously loaded context
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    psiflow_config, definitions = ExecutionContextLoader.parse_config(config_dict)
    return ExecutionContextLoader.load(psiflow_config, definitions)
