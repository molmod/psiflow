import warnings
import logging
from typing import Optional, Union, Callable, Optional
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
from psiflow.data.utils import extract_quantities
from psiflow.geometry import Geometry, NullState
from psiflow.utils.apps import copy_app_future
from psiflow.utils.parse import LineNotFoundError, get_task_name_id


logger = logging.getLogger(__name__)  # logging per module


class Status(Enum):
    SUCCESS = 0
    FAILED = 1
    INCONSISTENT = 2


def update_geometry(geom: Geometry, data: dict) -> Geometry:
    """"""
    # data should contain 'stdout' and 'status' keys
    task_name, task_id = get_task_name_id(data["stdout"])
    logger.info(f'Task "{task_name}" (ID {task_id}): {data["status"].name}')

    geom = geom.copy()
    geom.reset()
    geom.order["status"], geom.order["task_id"] = data["status"].name, task_id
    if data["status"] != Status.SUCCESS:
        return geom

    # necessary keys: 'positions' and 'energy'
    # optional keys: 'runtime' and 'forces'

    shift = data["positions"][0] - geom.per_atom.positions[0]
    if not np.allclose(data["positions"], geom.per_atom.positions + shift, atol=1e-6):
        # output does not match geometry up to a translation
        geom.order["status"] = Status.INCONSISTENT
        return geom

    geom.energy = data["energy"]
    geom.order["runtime"] = data.get("runtime")
    if "forces" in data:
        geom.per_atom.forces[:] = data["forces"]
    return geom


@join_app
def get_minimum_energy(element: str, **kwargs) -> AppFuture[float]:
    energies = {m: state.energy or np.inf for m, state in kwargs.items()}
    energy = min(energies.values())
    msg = [
        f"\nAtomic energies for element {element}",
        *[f"\tMultiplicity {m}:{energy:>10.4f} eV" for m, energy in energies.items()],
    ]
    logger.info('\n'.join(msg))
    assert not np.isinf(energy), f"Atomic energy calculation of '{element}' failed"
    return copy_app_future(energy)  # TODO: why?


def get_spin_multiplicities(element: str) -> list[int]:
    """TODO: rethink this"""
    # max S = N * 1/2, max mult = 2 * S + 1
    from ase.symbols import atomic_numbers

    mults = []
    number = atomic_numbers[element]
    for mult in range(1, min(number + 2, 16)):
        if number % 2 == 0 and mult % 2 == 0:
            continue  # S always whole, mult never even
        mults.append(mult)
    return mults


@join_app
def compute_dataset(
    dataset: Dataset,
    length: int,
    reference: "Reference",
) -> list[AppFuture[Geometry]]:
    logger.info(f"Performing {length} {reference.__class__.__name__} calculations.")
    geometries = dataset.geometries()  # read it once
    evaluated = [reference.evaluate(geometries[i]) for i in range(length)]
    return evaluated


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

    def compute(
        self,
        arg: Union[Dataset, Geometry, AppFuture, list],
        *outputs: Optional[Union[str, tuple]],
    ):
        # TODO: deprecate? Reference evaluations are too costly
        for output in outputs:
            if output not in self.outputs:
                raise ValueError("output {} not in {}".format(output, self.outputs))

        # TODO: convert_to_dataset util?
        if isinstance(arg, Dataset):
            dataset = arg
        elif isinstance(arg, list):
            dataset = Dataset(arg)
        elif isinstance(arg, AppFuture) or isinstance(arg, Geometry):
            dataset = Dataset([arg])
        else:
            raise TypeError

        # TODO: writing to dataset first is wasted overhead,
        #  but extract_quantities does not accept AppFuture[list[Geometry]] (yet?)
        future_dataset = self.compute_dataset(dataset)
        outputs = outputs or self.outputs
        future_data = extract_quantities(
            tuple(outputs), None, None, inputs=[future_dataset.extxyz]
        )
        if len(outputs) == 1:
            return future_data[0]
        return [future_data[_] for _ in range(len(outputs))]

    def compute_atomic_energy(self, element, box_size=None) -> AppFuture[float]:
        references = self.get_single_atom_references(element)
        state = Geometry.from_data(
            numbers=np.array([atomic_numbers[element]]),
            positions=np.array([[0, 0, 0]]),
            cell=np.eye(3) * (box_size or 0),
        )
        futures = {}
        for mult, reference in references.items():
            futures[str(mult)] = reference.evaluate(state)
        return get_minimum_energy(element, **futures)

    def get_execute_app(self) -> Callable:
        # TODO: how often is this called?
        context = psiflow.context()
        definition = context.definitions[self.executor]
        return partial(
            bash_app(_execute, executors=[self.executor]),
            bash_template=self.bash_template,
            parsl_resource_specification=definition.wq_resources(self.n_cores),
            label=self._execute_label,
        )

    def evaluate(self, geometry: Geometry | AppFuture) -> AppFuture:
        return evaluate(geometry, self)

    def compute_dataset(self, dataset: Dataset) -> Dataset:
        future = compute_dataset(dataset, dataset.length(), self)
        return Dataset(future)

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
        data = {"status": Status.FAILED}  # TODO: find out what went wrong?
    data |= {"stdout": Path(inputs[0]), "stderr": Path(inputs[1])}
    return update_geometry(geom, data)


process_output = python_app(_process_output, executors=["default_threads"])


@join_app
def evaluate(
    geom: Geometry,
    reference: Reference,
    execute_func: Optional[Callable] = None,
) -> AppFuture:
    """"""
    if geom == NullState:  # TODO: remove this
        warnings.warn("Skipping NullState..")
        return copy_app_future(geom)

    flag, *files = reference.create_input(geom=geom)
    if not flag:  # TODO: should we reset geom?
        return copy_app_future(geom)

    # do the actual evaluation
    if execute_func is None:
        execute_func = reference.get_execute_app()
    future = execute_func(inputs=files)

    future = process_output(
        geom=geom,
        reference=reference,
        inputs=[future.stdout, future.stderr, future],  # wait for future
    )
    return future
