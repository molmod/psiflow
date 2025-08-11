from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import ClassVar, Optional, Union, Callable
from pathlib import Path
from functools import partial
from enum import Enum

import numpy as np
import parsl
from ase.data import atomic_numbers
from parsl import python_app, join_app, bash_app, File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Computable, Dataset
from psiflow.data.utils import extract_quantities
from psiflow.geometry import Geometry
from psiflow.utils.apps import copy_app_future
from psiflow.utils.parse import LineNotFoundError


logger = logging.getLogger(__name__)  # logging per module

# TODO: fix tests + orca test
# TODO: some actual logging?
# TODO: safe_compute_dataset functionality?


class Status(Enum):
    SUCCESS = 0
    FAILED = 1
    INCONSISTENT = 2


def update_geometry(geom: Geometry, data: dict) -> Geometry:
    """"""
    geom = geom.copy()
    geom.reset()
    metadata = {k: data[k] for k in ("status", "stdout", "stderr", "exitcode")}
    geom.order |= metadata
    print(metadata)  # TODO: nice for debugging

    if data["status"] != Status.SUCCESS:
        return geom
    geom.order["runtime"] = data.get("runtime")

    shift = data["positions"][0] - geom.per_atom.positions[0]
    if not np.allclose(data["positions"], geom.per_atom.positions + shift, atol=1e-6):
        # output does not match geometry up to a translation
        geom.order["status"] = Status.INCONSISTENT
        return geom

    geom.energy = data["energy"]
    if "forces" in data:
        geom.per_atom.forces[:] = data["forces"]
    return geom


@join_app
def get_minimum_energy(element: str, **kwargs) -> AppFuture[float]:
    energies = {m: state.energy or np.inf for m, state in kwargs.items()}
    energy = min(energies.values())

    logger.info(f"Atomic energies for element {element}")
    for m, energy in energies.items():
        logger.info(f"\tMultiplicity {m}:{energy:>10.4f} eV")
    assert not np.isinf(energy), f"Atomic energy calculation of '{element}' failed"
    return copy_app_future(energy)


@join_app
def compute_dataset(
    dataset: Dataset,
    length: int,
    reference: Reference,
) -> list[AppFuture[Geometry]]:
    logger.info(f"Performing {length} {reference.__class__.__name__} calculations.")
    geometries = dataset.geometries()  # read it once
    evaluated = [reference.evaluate(geometries[i]) for i in range(length)]
    return evaluated


def _execute(
    reference: Reference,
    inputs: list[File],
    parsl_resource_specification: Optional[dict] = None,
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
    label: str = "singlepoint",
) -> str:
    return reference.get_shell_command(inputs)


def _process_output(
    reference: Reference,
    geom: Geometry,
    inputs: tuple[str | int] = (),
) -> Geometry:
    """"""
    stdout = Path(inputs[0]).read_text()
    try:
        data = reference.parse_output(stdout)
    except LineNotFoundError:
        # TODO: find out what went wrong
        data = {"status": Status.FAILED}
    data |= {
        "stdout": Path(inputs[0]).name,
        "stderr": Path(inputs[1]).name,
    }
    return update_geometry(geom, data)


@join_app
def evaluate(reference: Reference, geom: Geometry) -> AppFuture[Geometry]:
    """"""
    execute, *files = reference.app_pre(geom=geom)
    if not execute:  # TODO: should we reset geom?
        return copy_app_future(geom)
    future = reference.app_execute(inputs=files)
    future = reference.app_post(
        geom=geom, inputs=[future.stdout, future.stderr, future]  # wait for future
    )
    return future


@psiflow.serializable
class Reference(Computable):
    outputs: tuple[str, ...]
    batch_size: ClassVar[int] = 1  # TODO: not really used
    executor: str
    app_pre: ClassVar[Callable]  # TODO: fix serialisation
    app_execute: ClassVar[Callable]
    app_post: ClassVar[Callable]
    _execute_label: ClassVar[str]
    execute_command: str

    def compute(
        self,
        arg: Union[Dataset, Geometry, AppFuture, list],
        *outputs: Optional[Union[str, tuple]],
    ):
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

    def compute_dataset(self, dataset: Dataset) -> Dataset:
        future = compute_dataset(dataset, dataset.length(), self)
        return Dataset(future)

    def _create_apps(self):
        definition = psiflow.context().definitions[self.executor]
        self.execute_command = definition.command()
        wq_resources = definition.wq_resources()
        self.app_pre = self.create_input
        self.app_execute = partial(
            bash_app(_execute, executors=[self.executor]),
            reference=self,
            parsl_resource_specification=wq_resources,
            label=self._execute_label,
        )
        self.app_post = partial(
            python_app(_process_output, executors=["default_threads"]),
            reference=self,
        )

    def evaluate(self, geometry: Geometry | AppFuture) -> AppFuture:
        return evaluate(self, geometry)

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

    def get_single_atom_references(self, element: str) -> dict[int, Reference]:
        raise NotImplementedError

    def get_shell_command(self, inputs: list[File]) -> str:
        raise NotImplementedError

    def parse_output(self, stdout: str):
        raise NotImplementedError

    def create_input(self, geom: Geometry) -> tuple[bool, File, ...]:
        raise NotImplementedError


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
