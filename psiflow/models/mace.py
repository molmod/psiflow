import shutil
import tempfile
import logging
from pathlib import Path
from enum import StrEnum
from typing import Optional, Callable, Any
from collections.abc import Sequence
from functools import partial

import ase
import ase.io
import yaml
import parsl
from ase.data import atomic_numbers
from parsl import bash_app, python_app, File, join_app
from parsl.dataflow.futures import AppFuture, DataFuture, Future

import psiflow
from psiflow.data import Dataset
from psiflow.hamiltonians import MACEHamiltonian
from psiflow.utils.future import resolve_nested_futures
from psiflow.utils.parse import format_env_vars

# TODO: when changing the training dataset, the models avg_num_neighbors will also change,
#  making old checkpoints inconsistent
# TODO: check whether atomic energies are actually correct in the model
# TODO: sanitise_config consistently called?

logger = logging.getLogger(__name__)  # logging per module


KEY_ATOMIC_ENERGIES = "psiflow_atomic_energies"


class Status(StrEnum):
    NEW = "new"
    INIT = "init"
    TRAINED = "trained"


def sanitise_config(config: dict) -> None:
    """"""
    config["save_cpu"] = True  # save to CPU after training

    # keep all files in work_dir
    config["work_dir"] = "."
    keys = ("log_dir", "model_dir", "checkpoints_dir", "results_dir", "downloads_dir")
    for k in keys:
        config.pop(k, None)


def create_checkpoint_name(config: dict) -> str:
    name, seed = config["name"], config["seed"]
    return f"{name}_run-{seed}_epoch-0.pt"


def format_E0s(atomic_energies: dict) -> dict | str:
    """"""
    # TODO: MACE fails when atomic energy is not defined for every element..
    if atomic_energies:
        return {atomic_numbers[k]: v for k, v in atomic_energies.items()}
    else:
        return "average"


def get_bash_command(root: Path, pre: Optional[str] = None) -> str:
    defaults = [
        "mkdir checkpoints",  # otherwise MACE borks
        "mace_run_train --config {}",  # run MACE
        f"rsync -av --ignore-existing --exclude=/*.model ./ {root}/",  # copy things back
    ]
    if pre:
        defaults.insert(1, pre)
    return "\n".join(defaults)


def _execute(
    bash_template: str,
    inputs: list[File],
    parsl_resource_specification: Optional[dict] = None,
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
    label: str = "MACE",
) -> str:
    return bash_template.format(*inputs)


class MACEModel:
    """"""

    root: Path
    config: dict[str, Any]
    model_future: Optional[psiflow._DataFuture]
    atomic_energies: dict[str, float | AppFuture]

    def __init__(self, root: Path, config: Optional[dict] = None):
        self.root = psiflow.resolve_and_check(root)
        self.atomic_energies = {}
        self.config = config
        if config is not None:
            sanitise_config(config)
            yaml.safe_dump(config, self.path_config.open("w"))
        else:
            self._load_config()

        # create a lockfile to signal this root directory is in use
        # TODO: might need closing in __del__ of this class
        lock = tempfile.NamedTemporaryFile(prefix="psiflow-", suffix=".lock", dir=root)
        self.lock = lock

        self.model_future = None
        if (p := self.path_mlp).is_file():
            self.model_future = p

    def set_kwargs(self, **kwargs: Any | Future) -> None:
        """Update config arguments, possibly with futures"""
        self.config |= kwargs

    def train(self, training: Dataset, validation: Dataset) -> AppFuture:
        if self.status == Status.NEW:
            logger.warning("Attempting to train new model. Initialising first..")
            self.initialize(training)

        future_cfg = self._resolve_futures()
        future = train_app(
            self,
            future_cfg,
            training.extxyz,
            validation.extxyz,
            inputs=[self.model_future],  # wait for previous model training
        )
        outputs = [File(self.path_mlp), File(self.path_chkpt)]
        future = store_last(self, future, outputs=outputs)
        self.model_future = future.outputs[0]
        return future

    def initialize(self, dataset: Dataset) -> AppFuture:
        """Create and save the model architecture"""
        assert self.status == Status.NEW, "Already initialized.."
        future_cfg = self._resolve_futures()
        future = initialize_app(self, future_cfg, dataset.extxyz)
        future = store_last(self, future, outputs=[File(self.path_mlp)])
        self.model_future = future.outputs[0]
        return future

    def add_atomic_energy(self, element: str, energy: float | AppFuture) -> None:
        assert (
            self.status == Status.NEW
        ), "cannot add atomic energies after model has been initialized.."
        if element in self.atomic_energies:
            logger.warning(f"Overwriting existing atomic energy for '{element}'..")
        self.atomic_energies[element] = energy

    # def create_hamiltonian(self) -> MACEHamiltonian:
    #     assert self.status != Status.NEW, "Trained model does not exist.."
    #     # TODO: can still contain futures - atomic energies no longer necessary?
    #     return MACEHamiltonian(self.model_future, self.atomic_energies)

    def _load_config(self) -> None:
        """Does not care for futures"""
        config = yaml.safe_load(self.path_config.open())
        self.atomic_energies = config.pop(KEY_ATOMIC_ENERGIES, {})
        sanitise_config(config)
        self.config = config

    def _resolve_futures(self) -> AppFuture:
        """Wait for all futures in config and atomic energies"""
        cfg = self.config | {KEY_ATOMIC_ENERGIES: self.atomic_energies}
        return resolve_nested_futures(cfg)

    @staticmethod
    def get_app(init: bool = False) -> tuple[Callable, str]:
        # TODO: update -- why staticmethod and not function?
        context = psiflow.context()
        definition = context.definitions["ModelTraining"]
        template = context.bash_template
        env_vars = format_env_vars(definition.env_vars)
        app = partial(
            bash_app(_execute, executors=["ModelTraining"]),
            parsl_resource_specification=definition.wq_resources(),
            label="mace_init" if init else "mace_train",
        )
        return app, template.format(commands="{commands}", env=env_vars)

    @property
    def path_config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def path_mlp(self) -> Path:
        return self.root / "last.model"

    @property
    def path_chkpt(self) -> Path:
        return self.root / "last.pt"

    @property
    def path_checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def status(self) -> Status:
        if self.model_future is None:
            return Status.NEW
        if self.path_chkpt.is_file():
            return Status.TRAINED
        return Status.INIT

    @property
    def stage(self) -> str:
        # TODO: very ugly
        iteration = -1
        files = list(self.path_checkpoints.glob("*.pt"))
        if files:
            iteration = sorted(files)[-1].name.split("_")[0].split("-")[-1]
        return f"train-{int(iteration) + 1}"

    @classmethod
    def create(cls, path_dir: Path, config: dict):
        """Create a new model in a fresh directory"""
        (path := Path(path_dir)).mkdir()
        return cls(path, config)

    @classmethod
    def load(cls, path_dir: Path):
        """Load model from existing directory"""
        path_dir = Path(path_dir)
        assert len(list(path_dir.glob("*.lock"))) == 0, "Model directory in use"
        return cls(path_dir)


@join_app
def initialize_app(model: MACEModel, config: dict, file_data: File) -> AppFuture:
    """"""
    # TODO: we can kill mace_run_train as soon as 'RESULTS' block starts
    logging.info("Initialising MACE model...")
    yaml.safe_dump(config, model.path_config.open("w"))

    # make dummy val set
    file_val = psiflow.context().new_file("dummy_", ".xyz")
    atoms = ase.io.read(file_data.filepath)
    dummy = ase.Atoms(numbers=atoms.numbers[:1])
    ase.io.write(file_val.filepath, dummy)

    # update train config
    config |= {
        "name": "init",
        "max_num_epochs": 0,
        "train_file": file_data.filepath,
        "valid_file": file_val.filepath,
    }
    atomic_energies = config.pop(KEY_ATOMIC_ENERGIES)
    config["E0s"] = format_E0s(atomic_energies)
    file_cfg = psiflow.context().new_file("mace_cfg_", ".yaml")
    yaml.safe_dump(config, open(file_cfg.filepath, "w"))

    command = get_bash_command(model.root)
    app, template = model.get_app(init=True)
    return app(template.format(commands=command), inputs=[file_cfg])


@join_app
def train_app(
    model: MACEModel, config: dict, file_train: File, file_val, inputs: Sequence = ()
) -> AppFuture:
    """Wait for inputs and (re)train model"""
    if model.status == Status.TRAINED:
        logging.info("Retraining MACE model...")
    else:
        logging.info("Training initialised MACE model...")
    yaml.safe_dump(config, model.path_config.open("w"))

    # update train config
    config |= {
        "name": model.stage,
        "train_file": file_train.filepath,
        "valid_file": file_val.filepath,
    }
    atomic_energies = config.pop(KEY_ATOMIC_ENERGIES)
    config["E0s"] = format_E0s(atomic_energies)
    file_cfg = psiflow.context().new_file("mace_cfg_", ".yaml")
    yaml.safe_dump(config, open(file_cfg.filepath, "w"))

    pre = None
    if model.status == Status.TRAINED:
        chkpt_out = f"checkpoints/{create_checkpoint_name(config)}"
        pre = f"cp {model.path_chkpt} {chkpt_out}"
    command = get_bash_command(model.root, pre=pre)
    app, template = model.get_app(init=False)
    return app(template.format(commands=command), inputs=[file_cfg])


@python_app(executors=["default_threads"])
def store_last(model: MACEModel, future: Any, outputs: Sequence[File] = ()) -> None:
    """Waits for future and copies last MACE model (and checkpoint)"""
    files = list(model.path_checkpoints.glob("*.model"))
    shutil.copy2(sorted(files)[-1], outputs[0].filepath)
    if len(outputs) == 1:
        return  # only copy the model

    # also copy the checkpoint
    files = list(model.path_checkpoints.glob("*.pt"))
    shutil.copy2(sorted(files)[-1], outputs[1].filepath)
