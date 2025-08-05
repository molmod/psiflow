from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import ClassVar, Optional, Union

import numpy as np
from ase.data import atomic_numbers
from parsl.app.app import join_app, python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Computable, Dataset
from psiflow.data.utils import extract_quantities
from psiflow.geometry import Geometry
from psiflow.utils.apps import copy_app_future

logger = logging.getLogger(__name__)  # logging per module

# TODO: cleanup flow of apps
# TODO: remove typeguard
# TODO: we compute geoms -> extract data -> insert data again.. this is a bit stupid
# TODO: use 'geom' consistently
# TODO: _process_output might be general for all
# TODO: error handling
# TODO: fix tests
# TODO: task naming is no longer descriptive
# TODO: some actual logging?


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
) -> list[AppFuture]:
    logger.info(f"Performing {length} {reference.__class__.__name__} calculations.")
    geometries = dataset.geometries()  # read it once
    evaluated = [reference.evaluate(geometries[i]) for i in range(length)]
    return evaluated


@psiflow.serializable
class Reference(Computable):
    outputs: tuple[str, ...]
    batch_size: ClassVar[int] = 1  # not really used

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
        future_geoms = compute_dataset(dataset, dataset.length(), self)

        outputs = outputs or self.outputs
        future_data = extract_quantities(tuple(outputs), None, None, *future_geoms)
        if len(outputs) == 1:
            return future_data[0]
        return [future_data[_] for _ in range(len(outputs))]

    def compute_dataset(self, dataset: Dataset) -> Dataset:
        futures = compute_dataset(dataset, dataset.length(), self)
        return Dataset(futures)

    def evaluate(self, geometry: Geometry | AppFuture) -> AppFuture:
        # Every subclass needs a unique implementation
        raise NotImplementedError

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
