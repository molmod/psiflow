import logging
from typing import Callable, Optional
from collections.abc import Sequence
from pathlib import Path
from functools import partial
from enum import Enum

import numpy as np
import parsl
from ase.data import atomic_numbers
from parsl import python_app, join_app, bash_app, File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry
from psiflow.utils.apps import copy_app_future
from psiflow.utils.parse import LineNotFoundError, get_task_name_id


# TODO: existing Reference instances do not react to changes in execution environment
#  e.g., altering max_runtime is not retroactively updated


logger = logging.getLogger(__name__)  # logging per module


class Status(Enum):
    SUCCESS = 0
    FAILED = 1
    INCONSISTENT = 2


def update_geometry(geom: Geometry, data: dict) -> Geometry:
    """"""
    # TODO: is the metadata not more a logging thing?
    # data should contain 'stdout' and 'status' keys
    task_name, task_id = get_task_name_id(data["stdout"])
    logger.info(f'Task "{task_name}" (ID {task_id}): {data["status"].name}')

    geom = geom.copy()
    geom.reset()
    geom.metadata = {'status': data["status"].name, 'task_id': task_id}
    if data["status"] != Status.SUCCESS:
        return geom

    # necessary keys: 'positions' and 'energy'
    # optional keys: 'runtime' and 'forces'

    shift = data["positions"][0] - geom.per_atom.positions[0]
    if not np.allclose(data["positions"], geom.per_atom.positions + shift, atol=1e-6):
        # output does not match geometry up to a translation
        geom.status = Status.INCONSISTENT
        return geom

    geom.energy = data["energy"]
    geom.metadata["runtime"] = data.get("runtime")
    if "forces" in data:
        geom.per_atom.forces = data["forces"]
    return geom


@python_app(executors=['default_threads'])
def get_minimum_energy(element: str, **kwargs) -> AppFuture:
    energies = {m: state.energy or np.inf for m, state in kwargs.items()}
    energy = min(energies.values())
    msg = [
        f"\nAtomic energies for element {element}",
        *[f"\tMultiplicity {m}:{energy:>10.4f} eV" for m, energy in energies.items()],
    ]
    logger.info('\n'.join(msg))
    assert not np.isinf(energy), f"Atomic energy calculation of '{element}' failed"
    return energy


def get_spin_multiplicities(element: str) -> list[int]:
    """TODO: redo this"""
    # max S = N * 1/2, max mult = 2 * S + 1
    from ase.symbols import atomic_numbers

    mults = []
    number = atomic_numbers[element]
    for mult in range(1, min(number + 2, 16)):
        if number % 2 == 0 and mult % 2 == 0:
            continue  # S always whole, mult never even
        mults.append(mult)
    return mults


def _execute(
    bash_template: str,
    inputs: list[File],
    parsl_resource_specification: Optional[dict] = None,
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
    label: str = "singlepoint",
) -> str:
    return bash_template.format(*inputs)


class Reference:
    outputs: tuple[str, ...]
    bash_template: str
    _execute_label: str
    executor: str
    n_cores: Optional[int]

    def __init__(
        self, outputs: Sequence[str] = ("energy", "forces"), n_cores: int | None = None
    ):
        self.outputs = tuple(outputs)
        self.n_cores = n_cores

        context = psiflow.context()
        definition = context.definitions[self.executor]
        if (n := self.n_cores) is not None:
            assert n <= definition.spec["cores"]

    def __call__(self, geometry: Geometry | AppFuture) -> AppFuture:
        return evaluate(geometry, self)

    def evaluate(self, dataset: Dataset) -> Dataset:
        """Return a new labelled dataset"""
        future = evaluate_geometries(dataset.geometries(), self)
        return Dataset(future)

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture:
        references = self.get_single_atom_references(element)
        state = Geometry.from_data(
            numbers=np.array([atomic_numbers[element]]),
            positions=np.array([[0, 0, 0]]),
            cell=np.eye(3) * (box_size or 0),
        )
        futures = {}
        for mult, reference in references.items():
            futures[str(mult)] = reference(state)
        return get_minimum_energy(element, **futures)  # float

    def get_execute_app(self) -> Callable:
        context = psiflow.context()
        definition = context.definitions[self.executor]
        return partial(
            bash_app(_execute, executors=[self.executor]),
            bash_template=self.bash_template,
            parsl_resource_specification=definition.wq_resources(self.n_cores),
            label=self._execute_label,
        )

    def get_single_atom_references(self, element: str) -> dict[int, "Reference"]:
        raise NotImplementedError

    def create_input(self, geom: Geometry) -> tuple[bool, File, ...]:
        raise NotImplementedError

    def parse_output(self, stdout: str) -> dict:
        raise NotImplementedError


def _process_output(
    geom: Geometry,
    reference: Reference,
    inputs: Sequence[File] = (),
) -> Geometry:
    """Updates geometry with ab initio labels"""
    stdout = Path(inputs[0]).read_text()
    try:
        data = reference.parse_output(stdout)
    except LineNotFoundError:
        data = {"status": Status.FAILED}
    data |= {"stdout": Path(inputs[0]), "stderr": Path(inputs[1])}
    return update_geometry(geom, data)


process_output = python_app(_process_output, executors=["default_threads"])


@join_app
def evaluate(
    geom: Geometry,
    reference: Reference,
    execute_app: Optional[Callable] = None,
) -> AppFuture:
    """"""
    flag, *files = reference.create_input(geom=geom)
    if not flag:
        geom.reset()  # remove fields to indicate no evaluation happened
        return copy_app_future(geom)

    # do the actual evaluation
    if execute_app is None:
        execute_app = reference.get_execute_app()
    future = execute_app(inputs=files)

    future = process_output(
        geom=geom,
        reference=reference,
        inputs=[future.stdout, future.stderr, future],  # wait for future
    )
    return future


@join_app
def evaluate_geometries(
    states: Sequence[Geometry],
    reference: "Reference",
) -> list[AppFuture]:
    msg = f"Performing {len(states)} {reference.__class__.__name__} calculations."
    logger.info(msg)
    execute_app = reference.get_execute_app()
    evaluated = [evaluate(geom, reference, execute_app) for geom in states]
    return evaluated
