from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import ClassVar, Optional, Union, Callable
from pathlib import Path
from functools import partial

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
from psiflow.reference.utils import (
    Status,
    copy_data_to_geometry,
)

logger = logging.getLogger(__name__)  # logging per module

# TODO: cleanup flow of apps
# TODO: remove typeguard
# TODO: we compute geoms -> extract data -> insert data again.. this is a bit stupid
# TODO: use 'geom' consistently
# TODO: _process_output might be general for all
# TODO: error handling
# TODO: fix tests + orca test
# TODO: task naming is no longer descriptive
# TODO: some actual logging?
# TODO: stuff some init things into base class
# TODO: make every reference subclass have its own 'execute command' + extract _execute
# TODO: safe_compute dataset functionality?
# TODO: fix GPAW path import issues..?
# TODO: GPAW make overview of acceptable params
# TODO: raise error in find_line


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
    with open(inputs[0], "r") as f:
        stdout = f.read()
    try:
        data = reference.parse_output(stdout)
    except TypeError:
        # TODO: find out what went wrong
        data = {"status": Status.FAILED}
    data |= {
        "stdout": Path(inputs[0]).name,
        "stderr": Path(inputs[1]).name,
        "exitcode": inputs[2],  # TODO: will we reach this point if bash app failed?
    }
    return copy_data_to_geometry(geom, data)


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
    _execute_label: ClassVar[str]  # TODO: hhmm?
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
