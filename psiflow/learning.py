import shutil
import logging
import warnings
from pathlib import Path
from typing import Optional, Any, Protocol
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import tabulate
from parsl import File, python_app, join_app
from parsl.dataflow.futures import AppFuture, Future

import psiflow
from psiflow.geometry import Geometry, MISSING
from psiflow.data import Dataset, read_frames, write_frames
from psiflow.data.utils import assign_ids
from psiflow.hamiltonians import Hamiltonian, MixtureHamiltonian
from psiflow.models import MACE
from psiflow.reference import Reference
from psiflow.sampling import SimulationOutput, Walker, sample
from psiflow.sampling.output import Status
from psiflow.compute import _compare_arrays
from psiflow.serialization import _deserialize
from psiflow.utils.apps import pack, copy_data_future, log_message
from psiflow.utils.io import _dump_json

logger = logging.getLogger(__name__)  # logging per module


FILE_SAMPLE = "0_sampled.xyz"
FILE_REFERENCE = "1_evaluated.xyz"
FILE_DATA = "2_data.xyz"
FILE_TRAIN = "train.xyz"
FILE_VAL = "val.xyz"
FILE_MODEL = "potential.model"
FILE_LEARNING = "learning.json"
FILE_RECORDS = "records.json"
FILE_WALKERS = "walkers.json"


# TODO: some plugin for data selection / uncertainty thing
# TODO: some plugin for wandb?
# TODO: clean_data method to throw away ridiculous structures?
# TODO: log final errors of entire dataset?


@psiflow.register_serializable
@dataclass
class Record:
    identifier: int
    walker: int
    e_rmse: float
    f_rmse: float
    reset: bool
    discard: bool


@psiflow.register_serializable
@dataclass
class WalkerRecord:
    id: int
    status: str
    temp: float
    time: float
    reset: bool


class Thresholds(Protocol):
    def __call__(self, geom: Geometry, record: Record) -> Record: ...


class Metrics(Protocol):
    def __call__(self, root: Path, workdir: Path) -> None: ...


class Learning:
    model: MACE
    reference: Reference
    train_valid_split: float
    identifier: AppFuture | int
    thresholds: Optional[Thresholds]
    metrics: list[Metrics]
    iteration: int
    workdir: Optional[Path]
    wait_for: Optional[Future]
    futures: list[Future]

    def __init__(
        self,
        model: MACE,
        reference: Reference,
        train_valid_split: float = 0.9,
        initial_data: Optional[Dataset] = None,
    ):
        self.model = model
        self.reference = reference
        self.train_valid_split = train_valid_split
        self.thresholds = None
        self.metrics = [RecordLogger()]

        self.iteration = -1
        self.workdir = None
        self.wait_for = None  # wait for previous learning step
        self.futures = []  # wait for full learning iteration

        if self._attempt_restart():
            return

        if initial_data is None:
            self.identifier = 0
        else:
            self.identifier = initial_data.assign_identifiers()
            initial_data.save(self.root / "initial_data.xyz")
            future = store_train_val(self.root, initial_data, self.train_valid_split)
            self.wait_for = future

    @property
    def root(self) -> Path:
        return self.model.root

    def add_thresholds(self, thresholds: Thresholds) -> None:
        self.thresholds = thresholds

    def add_post_metrics(self, metrics: Metrics) -> None:
        self.metrics.append(metrics)

    def passive_learning(
        self,
        hamiltonian: Hamiltonian,
        walkers: Sequence[Walker],
        steps: int,
        **sampling_kwargs,
    ) -> Sequence[Walker]:
        """Perform a passive learning iteration, sampling structures with the provided hamiltonian"""
        execute = self._next_iteration("passive_learning")
        if not execute:
            return self._load_walkers()

        data, outputs = self._sample(
            hamiltonian, walkers, steps=steps, **sampling_kwargs
        )
        records = self._evaluate(data, hamiltonian)
        self._analyze(walkers, outputs, records)
        self._train()
        self._save()
        self._post_process()

        return walkers

    def active_learning(
        self,
        walkers: Sequence[Walker],
        steps: int,
        reset_training: bool = False,
        **sampling_kwargs,
    ) -> Sequence[Walker]:
        """Perform a passive learning iteration, sampling structures with the model hamiltonian"""
        execute = self._next_iteration("active_learning")
        if not execute:
            return self._load_walkers()

        hamiltonian = self.model.create_hamiltonian()
        data, outputs = self._sample(
            hamiltonian, walkers, steps=steps, **sampling_kwargs
        )
        records = self._evaluate(data, hamiltonian)
        self._analyze(walkers, outputs, records)
        if reset_training:
            self._reset_training()
        self._train()
        self._save()
        self._post_process()

        return walkers

    def _sample(
        self, hamiltonian: Hamiltonian, walkers: Sequence[Walker], **kwargs
    ) -> tuple[Dataset, list[SimulationOutput]]:
        """Sample new structures and appropriately label them"""

        msg = "Learning hamiltonian is already part of the walker PES. Are you sure?"
        for w in walkers:
            h = w.hamiltonian
            flag = (
                isinstance(h, MixtureHamiltonian)
                and h.get_coefficient(hamiltonian) != 0
            )
            if h == hamiltonian or flag:
                warnings.warn(msg)

        kwargs["keep_trajectory"] = True
        if (step := kwargs.get("step")) is not None:
            # do not store initial structure
            kwargs.setdefault("start", step)

        # combine walker PES with learning hamiltonian
        backup = []
        for w in walkers:
            backup.append(w.hamiltonian)
            w.hamiltonian = w.hamiltonian + hamiltonian

        outputs = sample(walkers, **kwargs)

        # restore original walker hamiltonians
        for w, h in zip(walkers, backup):
            w.hamiltonian = h

        data: list[Future[Geometry | list[Geometry]]]
        if outputs[0].trajectory is not None:
            # depends on multiple sampling kwargs
            data = [out.trajectory.geometries() for out in outputs]
        else:
            data = [out.state for out in outputs]

        # label and store
        future = label_geometries(*data, identifier=self.identifier)
        states, self.identifier = future[0], future[1]
        file_data = File(self.workdir / FILE_SAMPLE)
        future_ = write_frames(states, outputs=[file_data])

        self.wait_for = future_
        self.futures += [future, future_]
        return Dataset(extxyz=future_.outputs[0]), outputs

    def _evaluate(self, data: Dataset, hamiltonian: Hamiltonian) -> Future:
        """Perform single point evaluations and filtering steps"""

        eval_ref = self.reference.evaluate(data)
        future0 = eval_ref.save(self.workdir / FILE_REFERENCE)
        eval_mod = hamiltonian.evaluate(data)
        future = compare_geometries(
            eval_ref.geometries(), eval_mod.geometries(), self.thresholds
        )
        data, records = future[0], future[1]

        file_data = File(self.workdir / FILE_DATA)
        future1 = write_frames(data, outputs=[file_data])
        data = Dataset(extxyz=future1.outputs[0])

        future2 = store_train_val(self.workdir, data, self.train_valid_split)
        self.wait_for = future2
        self.futures += [future0, records, future1, future2]
        return records  # Future[list[Record]]

    def _train(self) -> None:
        """Train model on all data and store"""
        future = collect_geometries(self.root, FILE_TRAIN, self.wait_for)
        train = Dataset(future)
        future = collect_geometries(self.root, FILE_VAL, self.wait_for)
        val = Dataset(future)
        self.model.train(train, val)

        file = File(self.workdir / FILE_MODEL)
        future = copy_data_future(self.model.model_future, outputs=[file])

        self.wait_for = self.model.wait_for
        self.futures += [future, self.model.wait_for]
        return

    def _analyze(
        self,
        walkers: Sequence[Walker],
        outputs: list[SimulationOutput],
        records: Future,
    ) -> None:
        """Analyse sampling and data output, resets walkers and stores"""
        future = analyze_outputs(
            records,
            pack(*[o.status for o in outputs]),
            pack(*[o.temperature for o in outputs]),
            pack(*[o.time for o in outputs]),
        )
        walker_records, reset_mask = future[0], future[1]
        for i, walker in enumerate(walkers):
            walker.conditional_reset(reset_mask[i])

        future_ = psiflow.serialize(
            list(walkers),
            path_json=self.workdir / FILE_WALKERS,
            copy_to=self.workdir,
        )
        file = File(self.workdir / FILE_RECORDS)
        future = store_records(records, walker_records, outputs=[file])

        self.futures += [walker_records, future_, future]
        return

    def _save(self) -> None:
        """Save attributes and state of this learning iteration"""
        keys = (
            "reference",
            "train_valid_split",
            "identifier",
            "thresholds",
            "metrics",
        )
        data = {k: v for k, v in vars(self).items() if k in keys}
        future = psiflow.serialize(
            data, self.workdir / FILE_LEARNING, copy_to=self.workdir
        )
        self.futures += [future]
        return

    def _post_process(self) -> None:
        """Optional post-processing steps after a complete learning iteration"""
        for metrics in self.metrics:
            post_process(metrics, self.root, self.workdir, inputs=self.futures)

    def _reset_training(self) -> None:
        """Remove trained model and recreate train/val splits"""
        self.model.reset()
        self.wait_for = recreate_train_val_splits(
            self.root, self.train_valid_split, self.wait_for
        )

    def _attempt_restart(self) -> bool:
        """Try to restart from the last fully completed learning iteration"""
        history = self._get_learning_history()
        completed = []
        for i, name in history.items():
            workdir = self.root / name
            if not check_learning_iteration(workdir):
                logger.warning(f"Removing partially finished iteration '{name}'..")
                shutil.rmtree(workdir)
            else:
                completed.append(i)

        if not completed:  # no completed iteration
            logger.info(f"Starting new Learning instance at '{self.root}'")
            self.model.model_future = None  # TODO: do we want to reset the model?
            return False

        iteration = max(completed)
        name = history[iteration]
        workdir = self.root / name
        data = _deserialize((workdir / FILE_LEARNING).read_text())
        self.identifier = data["identifier"]
        self.model.model_future = File(workdir / FILE_MODEL)
        logger.info(
            f"Reloading Learning instance at '{self.root}' from iteration {iteration}"
        )
        return True

    def _next_iteration(self, key: str) -> bool:
        """Prepare or skip the next learning iteration"""
        self.futures = []
        self.iteration += 1
        name = f"{self.iteration}_{key}"
        self.workdir = self.root / name

        history = self._get_learning_history()
        if self.iteration <= max(history.keys(), default=-1):
            if name in history.values():
                logger.info(f"Skipping iteration '{name}'")
            else:
                # TODO: should this only warn?
                msg = (
                    f"Refusing to execute iteration '{name}', which is inconsistent "
                    f"with history {list(history.values())}. Did you adapt the workflow?"
                )
                logger.warning(msg)
            return False

        log_message(
            logger.info,
            f"Executing learning iteration '{name}'",
            inputs=[self.wait_for],
        )
        self.workdir.mkdir()
        return True

    def _load_walkers(self) -> list[Walker]:
        file = self.workdir / FILE_WALKERS
        return _deserialize(file.read_text())

    def _get_learning_history(self) -> dict[int, str]:
        folders = [p for p in self.root.glob("*learning") if p.is_dir()]
        return {int(p.stem.split("_")[0]): p.stem for p in folders}


@python_app(executors=["default_threads"])
def label_geometries(
    *states: Geometry | Sequence[Geometry], identifier: int
) -> tuple[list[Geometry], int]:
    """Tag geometries with walker index and identifier"""
    data = []
    for i, state in enumerate(states):
        if isinstance(state, Geometry):
            state.walker_id = i
            data.append(state)
        else:
            for s in state:
                s.walker_id = i
            data.extend(state)

    _, identifier = assign_ids(data, identifier)
    return data, identifier


@python_app(executors=["default_threads"])
def compare_geometries(
    geoms_ref: Sequence[Geometry],
    geoms_mod: Sequence[Geometry],
    thresholds: Optional[Thresholds],
) -> tuple[list[Geometry], list[Record]]:
    """Create data records and (optionally) discard evaluated geometries"""
    records = []
    for geom_ref, geom_mod in zip(geoms_ref, geoms_mod):
        i, w = geom_ref.identifier, geom_ref.walker_id
        e_ref, e_mod = geom_ref.energy, geom_mod.energy
        f_ref, f_mod = geom_ref.per_atom.forces, geom_mod.per_atom.forces

        if e_ref is MISSING or f_ref is MISSING:
            # reference evaluation failed
            # TODO: what if reference only evaluates energies?
            record = Record(i, w, np.nan, np.nan, True, True)
        else:
            e_rmse = _compare_arrays(e_ref, e_mod)
            f_rmse = _compare_arrays(f_ref, f_mod)
            record = Record(i, w, e_rmse, f_rmse, False, False)

        if thresholds is not None:
            record = thresholds(geom_ref, record)

        records.append(record)

    geoms = [geom for geom, record in zip(geoms_ref, records) if not record.discard]
    return geoms, records


@python_app(executors=["default_threads"])
def collect_geometries(
    root: Path, pattern: str, wait_for: Any = None
) -> list[Geometry]:
    """Return all geometries stored under root found by glob in a deterministic order"""
    files = sorted(root.rglob(pattern))
    return [geom for f in files for geom in read_frames(f).result()]


@python_app(executors=["default_threads"])
def analyze_outputs(
    records: list[Record],
    statuses: list[Status],
    temperatures: list[float],
    times: list[float],
) -> tuple[list[WalkerRecord], list[bool]]:
    """Make walker records from sampling output and data records"""
    walkers_to_reset = set(r.walker for r in records if r.reset)
    records = []
    for i, (status, temp, time) in enumerate(zip(statuses, temperatures, times)):
        reset = status not in (Status.DONE, Status.TIMEOUT) or i in walkers_to_reset
        record = WalkerRecord(i, status.name, temp, time, reset)
        records.append(record)
    return records, [r.reset for r in records]


@python_app(executors=["default_threads"])
def store_records(
    records: list[Record], walker_records: list[WalkerRecord], outputs: list[File] = []
) -> None:
    assert len(outputs) == 1
    _dump_json(outputs=outputs, records=records, walker_records=walker_records)


def store_train_val(root: Path, data: Dataset, split: float) -> Future:
    """"""
    train, val = data.split(split)
    future_train = train.save(root / FILE_TRAIN)
    future_val = val.save(root / FILE_VAL)
    return pack(future_train, future_val)


@join_app
def recreate_train_val_splits(root: Path, split: float, wait_for: Any = None) -> Future:
    """"""
    logger.info("Resetting model and recreating train/val splits")
    data_files = sorted(root.rglob("*data.xyz"))
    train_files = sorted(root.rglob(FILE_TRAIN))
    val_files = sorted(root.rglob(FILE_VAL))
    assert len(data_files) == len(train_files) == len(val_files)

    for f in train_files + val_files:
        f.unlink()

    futures = []
    for f in data_files:
        future = store_train_val(f.parent, Dataset.load(f), split)
        futures.append(future)

    return futures


def check_learning_iteration(root: Path) -> bool:
    """Verifies iteration progress through file existence"""
    files = (
        FILE_SAMPLE,
        FILE_REFERENCE,
        FILE_DATA,
        FILE_TRAIN,
        FILE_VAL,
        FILE_MODEL,
        FILE_LEARNING,
        FILE_RECORDS,
        FILE_WALKERS,
    )
    return all((root / f).is_file() for f in files)


@join_app
def post_process(
    metrics: Metrics, root: Path, workdir: Path, inputs: list = []
) -> None:
    metrics(root, workdir)


@psiflow.register_serializable
@dataclass
class ErrorThresholds:
    """Reset and discard based on RMSE thresholds"""

    energy_reset: float = np.inf
    energy_discard: float = np.inf
    forces_reset: float = np.inf
    forces_discard: float = np.inf

    def __call__(self, geom: Geometry, record: Record) -> Record:
        if self.energy_reset < record.e_rmse or self.forces_reset < record.f_rmse:
            record.reset = True
        if self.energy_discard < record.e_rmse or self.forces_discard < record.f_rmse:
            record.discard = True
        return record


@psiflow.register_serializable
class RecordLogger:
    """Logs data and walker records to stdout"""

    def __init__(self) -> None:
        self.kwargs = dict(headers="keys", tablefmt="grid", floatfmt=".2f")

    def __call__(self, root: Path, workdir: Path) -> None:
        file = workdir / FILE_RECORDS
        records_dict = _deserialize(file.read_text())

        table = self.make_data_table(records_dict["records"])
        walker_table = self.make_walkers_table(records_dict["walker_records"])
        width = table.find("\n")
        title = f' OVERVIEW LEARNING ITERATION "{workdir.stem}" '
        text = "\n".join([title.center(width, "~"), table, walker_table])
        print(text)
        return

    def make_data_table(self, records: list[Record]) -> str:
        data = {
            "identifier": [r.identifier for r in records],
            "walker": [r.walker for r in records],
            "e_rmse [meV/atom]": np.array([r.e_rmse for r in records]) * 1000,
            "f_rmse [mev/Å]": np.array([r.f_rmse for r in records]) * 1000,
            "discard": [r.discard for r in records],
        }
        return tabulate.tabulate(data, **self.kwargs)

    def make_walkers_table(self, records: list[WalkerRecord]) -> str:
        data = {
            "walker": [r.id for r in records],
            "status": [r.status for r in records],
            "temperature [K]": [r.temp for r in records],
            "time [fs]": np.array([r.time for r in records]) * 1000,
            "reset": [r.reset for r in records],
        }
        return tabulate.tabulate(data, **self.kwargs)
