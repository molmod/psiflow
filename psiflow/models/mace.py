import shutil
import logging
import weakref
from pathlib import Path
from enum import StrEnum
from typing import Optional, Any

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
from psiflow.serialization import _DataFuture
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
    defaults = dict(energy_key="energy", forces_key="forces", seed=42)
    forced = dict(save_cpu=True, work_dir=".")
    cfg = defaults | config | forced

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


def iteration_from_model(path: Path) -> int:
    return int(path.stem.split('-')[0])


@bash_app(executors=["ModelTraining"])
def execute(
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
    wait_for: Optional[Future]

    def __init__(self, root: Path, config: Optional[dict] = None):
        # make sure nothing else is using the root directory
        assert str(root) not in MODEL_DIRS, "Model directory in use.."
        MODEL_DIRS[str(root)] = self

        self.root = Path(root)
        self.config = {}
        self.iteration = -1
        self.atomic_energies = {}
        self.model_future = None
        self.wait_for = None

        if self.path_config.is_file():
            self._load_config()
        if config is not None:
            self.config |= sanitise_config(config)
            yaml.safe_dump(self.full_config, self.path_config.open("w"))

        if (p := self._get_final_model()) is not None:
            assert self.iteration == iteration_from_model(p)
            self.model_future = File(p)

    def update_kwargs(self, **kwargs: Any | Future) -> None:
        """Update config arguments, possibly with futures"""
        self.config |= kwargs

    def train(self, training: Dataset, validation: Dataset) -> AppFuture:
        """Retrain the stored model from its old weights"""
        if not self.has_model_future:
            logger.warning("Attempting to train new model. Initialising first..")
            self._train(training.extxyz)

        future = self._train(training.extxyz, validation.extxyz)
        return future

    def initialize(self, dataset: Dataset) -> AppFuture:
        """Create and save the model architecture"""
        assert not self.has_model_future, "Already initialized.."
        return self._train(dataset.extxyz)

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
        return MACEHamiltonian(self.model_future)

    def reset(self) -> None:
        """Reset trained model to retrigger initialisation"""
        self.model_future = None

    def _train(
        self, path_train: _DataFuture, path_val: Optional[_DataFuture] = None
    ) -> AppFuture:
        """"""
        self.iteration += 1
        future = train_app(
            self,
            self._resolve_config_futures(),
            path_train,
            path_val,
            inputs=[self.model_future, self.wait_for],  # wait for model future and previous training run
            outputs=[psiflow.context().new_file("mace_", ".model")],
        )
        self.wait_for = future
        self.model_future = future.outputs[0]
        return future

    def _load_config(self) -> None:
        """"""
        config = yaml.safe_load(self.path_config.open())
        self.atomic_energies = config.pop(KEY_ATOMIC_ENERGIES)
        self.iteration = config.pop(KEY_ITERATION)
        self.config = sanitise_config(config)

    def _get_final_model(self) -> Optional[Path]:
        """Return the most recent model stored under checkpoints"""
        files = list(self.path_checkpoints.glob("*.model"))
        if len(files) == 0:
            return None
        return max(files, key=lambda p: iteration_from_model(p))

    def _resolve_config_futures(self) -> AppFuture:
        """Wait for all futures in config and atomic energies"""
        return resolve_nested_futures(self.full_config)

    @property
    def path_config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def path_checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def has_model_future(self) -> bool:
        # whether a model exists (or will exist)
        return self.model_future is not None

    @property
    def full_config(self) -> dict:
        return self.config | {
            KEY_ATOMIC_ENERGIES: self.atomic_energies,
            KEY_ITERATION: self.iteration,
        }

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
        assert path.is_dir()
        return cls(path)


@join_app
def train_app(
    model: MACE,
    config: dict,
    file_train: File,
    file_val: Optional[File] = None,
    inputs: list = [],
    outputs: list[File] = [],
) -> AppFuture:
    """Wait for inputs and (re)train model"""
    assert len(outputs) == 1
    assert (file_val is None) == (inputs[0] is None)  # correct model initialisation?
    initialisation = file_val is None
    config_back = config.copy()
    iteration = config.pop(KEY_ITERATION)

    if initialisation:
        # TODO: we can kill mace_run_train as soon as 'RESULTS' block starts
        logger.info(f"Initialising MACE model (iteration {iteration})...")
    else:
        logger.info(f"(Re)training MACE model (iteration {iteration})...")

    if initialisation:
        # make dummy val set
        file_val = psiflow.context().new_file("dummy_", ".xyz")
        atoms = ase.io.read(file_train.filepath)
        dummy = ase.Atoms(numbers=atoms.numbers[:1])
        ase.io.write(file_val.filepath, dummy)

    # update train config
    config |= {
        "name": f"{iteration}-{'init' if initialisation else 'train'}",
        "train_file": file_train.filepath,
        "valid_file": file_val.filepath,
        "E0s": format_E0s(config.pop(KEY_ATOMIC_ENERGIES)),
    }
    if initialisation:
        config["max_num_epochs"] = 0
    else:
        config["foundation_model"] = inputs[0].filepath  # restart from previous model

    config_back["name"] = config["name"]

    # execute bash app and post-process immediately
    # join_app unpacks futures in a way that loses stdout and stderr
    bash_future = execute_train_command(model.root, config)
    inputs = [bash_future.stdout, bash_future.stderr, bash_future]
    future: AppFuture = process_output(model, config_back, inputs=inputs, outputs=outputs)

    return future


def execute_train_command(root: Path, config: dict) -> AppFuture:
    """Prepare and run the bash app"""
    context = psiflow.context()
    definition = context.definitions["ModelTraining"]
    resources = definition.wq_resources()
    env_vars = format_env_vars(definition.env_vars)

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
        f"rsync -av --ignore-existing --exclude=/*.model ./ {root}/",  # copy things back
    ]
    command = "\n".join([l for l in command_lines if l])

    future = execute(
        bash_template=context.bash_template.format(commands=command, env=env_vars),
        inputs=[file],
        parsl_resource_specification=resources,
        label="mace-" + config["name"],
    )
    return future


@python_app(executors=["default_threads"])
def process_output(model: MACE, config: dict, inputs: list = [], outputs: list = []) -> None:
    """Waits for future and processes MLP training output"""

    # copy last model
    model_path = model._get_final_model()
    if model_path is not None and config['name'] in model_path.name:
        status = Status.SUCCESS
        shutil.copy2(model_path, outputs[0])
    else:
        status = Status.FAILURE

    name, task_id = get_task_name_id(inputs[0])
    if status == Status.SUCCESS:
        # only update stored config if training is successful
        logger.info(f"MACE training [ID {task_id}]: {status}")
        yaml.safe_dump(config, model.path_config.open("w"))
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
