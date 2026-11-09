import shutil
import logging
import weakref
from pathlib import Path
from enum import StrEnum
from typing import Optional, Any
from collections.abc import Sequence

import ase
import ase.io
import yaml
import parsl
from ase.data import atomic_numbers
from parsl import bash_app, python_app, File, join_app
from parsl.dataflow.futures import AppFuture, Future

import psiflow
from psiflow.data import Dataset
from psiflow.hamiltonians import MACEHamiltonian
from psiflow.utils.future import resolve_nested_futures
from psiflow.utils.parse import format_env_vars, get_task_name_id

# TODO: when changing the training dataset, the computed avg_num_neighbors will also change,
#  making old checkpoints inconsistent
# TODO: training fails when batch_size > dataset (see github issue)
# TODO: restart_latest functionality to continue training??

logger = logging.getLogger(__name__)  # logging per module


KEY_ATOMIC_ENERGIES = "psiflow_atomic_energies"
KEY_ITERATION = "psiflow_train_iteration"


MODEL_DIRS = weakref.WeakValueDictionary()


class Status(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    BAD_INPUT = "BAD INPUT"
    UNKNOWN_ERROR = "UNKNOWN ERROR"


def sanitise_config(config: dict) -> dict:
    """"""
    defaults = dict(
        energy_key="energy", forces_key="forces", seed=42, save_cpu=True, work_dir="."
    )
    cfg = defaults | config

    # keep all files in work_dir
    keys = ("log_dir", "model_dir", "checkpoints_dir", "results_dir", "downloads_dir")
    for k in keys:
        cfg.pop(k, None)

    return cfg


def format_E0s(atomic_energies: dict) -> dict | str:
    """MACE will fail if atomic energy is not defined for all elements"""
    if atomic_energies:
        return {atomic_numbers[k]: v for k, v in atomic_energies.items()}
    else:
        return "average"


def _execute(
    bash_template: str,
    inputs: list[File],
    parsl_resource_specification: Optional[dict] = None,
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
    label: str = "MACE",
) -> str:
    return bash_template.format(*inputs)


class MACE:
    """"""

    root: Path
    config: dict[str, Any]
    iteration: int
    model_future: Optional[psiflow._DataFuture]
    atomic_energies: dict[str, float | AppFuture]

    def __init__(self, root: Path, config: Optional[dict] = None):
        # make sure nothing else is using the root directory
        assert str(root) not in MODEL_DIRS, "Model directory in use.."
        MODEL_DIRS[str(root)] = self

        self.root = root
        self.iteration = -1
        self.atomic_energies = {}
        self.model_future = None

        if config is not None:
            self.config = sanitise_config(config)
            yaml.safe_dump(config, self.path_config.open("w"))
        else:
            self._load_config()
        if (p := self.path_mlp).is_file():
            self.model_future = File(p)

    def update_kwargs(self, **kwargs: Any | Future) -> None:
        """Update config arguments, possibly with futures"""
        self.config |= kwargs

    def train(self, training: Dataset, validation: Dataset) -> AppFuture:
        if not self.has_model_future:
            logger.warning("Attempting to train new model. Initialising first..")
            self.initialize(training)

        future_cfg = self._resolve_futures()
        future = train_app(
            self,
            future_cfg,
            training.extxyz,
            validation.extxyz,
            inputs=[self.model_future],  # wait for previous model training
            outputs=[File(self.path_mlp)],
        )
        self.model_future = future.outputs[0]
        return future

    def initialize(self, dataset: Dataset) -> AppFuture:
        """Create and save the model architecture"""
        assert not self.has_model_future, "Already initialized.."
        future_cfg = self._resolve_futures()
        future = initialize_app(
            self, future_cfg, dataset.extxyz, outputs=[File(self.path_mlp)]
        )
        self.model_future = future.outputs[0]
        return future

    def add_atomic_energy(self, element: str, energy: float | AppFuture) -> None:
        assert (
            not self.has_model_future
        ), "Cannot add atomic energies after model has been initialized.."
        if element in self.atomic_energies:
            logger.warning(f"Overwriting existing atomic energy for '{element}'..")
        self.atomic_energies[element] = energy

    def create_hamiltonian(self) -> MACEHamiltonian:
        # atomic energies are already part of the model
        assert self.has_model_future, "Trained model does not exist.."
        return MACEHamiltonian(self.model_future, {})

    def _load_config(self) -> None:
        """Does not care for futures"""
        config = yaml.safe_load(self.path_config.open())
        self.atomic_energies = config.pop(KEY_ATOMIC_ENERGIES, {})
        self.iteration = config.pop(KEY_ITERATION, 0)
        self.config = sanitise_config(config)

    def _resolve_futures(self) -> AppFuture:
        """Wait for all futures in config and atomic energies"""
        cfg = self.config | {KEY_ATOMIC_ENERGIES: self.atomic_energies}
        return resolve_nested_futures(cfg)

    def _execute_app(self, config: dict) -> AppFuture:
        """"""
        context = psiflow.context()
        definition = context.definitions["ModelTraining"]
        resources = definition.wq_resources()

        # final config tweaks
        if definition.multi_gpu:
            config["distributed"] = True
            config["launcher"] = "torchrun"
        else:
            config["distributed"] = False
        file = psiflow.context().new_file("mace_cfg_", ".yaml")
        yaml.safe_dump(config, open(file.filepath, "w"))

        # construct MACE train script
        command = "$(which mace_run_train) --config {}"
        if config["distributed"]:
            command = f"torchrun --standalone --nnodes=1 --nproc_per_node={resources['gpus']} {command}"
        command = definition.wrap_in_timeout(command)

        command_lines = [
            "mkdir checkpoints",  # otherwise MACE borks
            command,
            f"rsync -av --ignore-existing --exclude=/*.model ./ {self.root}/",  # copy things back
        ]
        command = "\n".join([l for l in command_lines if l])

        execute_app = bash_app(_execute, executors=["ModelTraining"])
        env_vars = format_env_vars(definition.env_vars)
        future = execute_app(
            bash_template=context.bash_template.format(commands=command, env=env_vars),
            inputs=[file],
            parsl_resource_specification=resources,
            label="mace_init" if config["name"] == "init" else "mace_train",
        )
        return future

    @property
    def path_config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def path_mlp(self) -> Path:
        return self.root / "last.model"

    @property
    def path_checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def has_model_future(self) -> bool:
        # whether a model exists (or will exist)
        return self.model_future is not None

    @classmethod
    def create(cls, path_dir: Path, config: dict):
        """Create a new model in a fresh directory"""
        path = psiflow.resolve_and_check(Path(path_dir))
        path.mkdir()
        return cls(path, config)

    @classmethod
    def load(cls, path_dir: Path):
        """Load model from existing directory"""
        path = psiflow.resolve_and_check(Path(path_dir))
        return cls(path)


@join_app
def initialize_app(
    model: MACE, config: dict, file_data: File, outputs: Sequence[File]
) -> AppFuture:
    """"""
    # TODO: we can kill mace_run_train as soon as 'RESULTS' block starts
    assert len(outputs) == 1
    logger.info("Initialising MACE model...")

    # make dummy val set
    file_val = psiflow.context().new_file("dummy_", ".xyz")
    atoms = ase.io.read(file_data.filepath)
    dummy = ase.Atoms(numbers=atoms.numbers[:1])
    ase.io.write(file_val.filepath, dummy)

    # update train config
    cfg = config.copy()
    cfg |= {
        "name": "init",
        "max_num_epochs": 0,
        "train_file": file_data.filepath,
        "valid_file": file_val.filepath,
    }
    cfg["E0s"] = format_E0s(cfg.pop(KEY_ATOMIC_ENERGIES))

    future = model._execute_app(cfg)
    inputs = [future.stdout, future.stderr, future]
    future_ = process_output(model, config, inputs=inputs)
    return future_


@join_app
def train_app(
    model: MACE,
    config: dict,
    file_train: File,
    file_val,
    inputs: Sequence = (),
    outputs: Sequence[File] = (),
) -> AppFuture:
    """Wait for inputs and (re)train model"""
    assert len(outputs) == 1
    if model.iteration == 0:
        logger.info("Training new MACE model...")
    else:
        logger.info("Retraining MACE model...")

    # update train config
    cfg = config.copy()
    cfg |= {
        "name": f"train-{model.iteration}",
        "train_file": file_train.filepath,
        "valid_file": file_val.filepath,
        "foundation_model": str(model.path_mlp),  # restart from previous model
    }
    cfg["E0s"] = format_E0s(cfg.pop(KEY_ATOMIC_ENERGIES))

    future = model._execute_app(cfg)
    inputs = [future.stdout, future.stderr, future]
    future_ = process_output(model, config, inputs=inputs)
    return future_


@python_app(executors=["default_threads"])
def process_output(model: MACE, config: dict, inputs: Sequence = ()) -> None:
    """Waits for future and processes MLP training output"""
    key = "init" if model.iteration < 0 else f"train-{model.iteration}"

    # copy last model
    files = list(model.path_checkpoints.glob(f"{key}*.model"))
    if len(files):
        status = Status.SUCCESS
        shutil.copy2(sorted(files)[-1], model.path_mlp)
    else:
        status = Status.FAILURE

    name, task_id = get_task_name_id(inputs[0])
    model.iteration += 1
    cfg = config | {KEY_ITERATION: model.iteration}
    if status == Status.SUCCESS:
        # only update stored config if training is successful
        logger.info(f"MACE training [ID {task_id}]: {status}")
        yaml.safe_dump(cfg, model.path_config.open("w"))
        return

    # check final error logs
    lines = Path(inputs[1]).read_text().rsplit(sep="\n", maxsplit=5)
    log = "\n".join(lines[1:])
    if "unrecognized arguments" in log:
        status = Status.BAD_INPUT
    else:
        status = Status.UNKNOWN_ERROR

    logger.warning(f"MACE training [ID {task_id}]: {status}")
    raise RuntimeError("MACE training failed. Check output logs.")


# copied from the MACE v0.3.15 repo
mace_mp_urls = {
    "small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "large": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model",
    "small-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model",
    "medium-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model",
    "small-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model",
    "medium-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model",
    "large-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model",
    "medium-0b3": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
    "medium-mpa-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "small-omat-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-small.model",
    "medium-omat-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "mace-matpes-pbe-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "mace-matpes-r2scan-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
    "mh-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
    "mh-1": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
}
