import math
from pathlib import Path
from typing import Callable, ClassVar, Optional, Union

import numpy as np
from parsl.app.app import join_app, python_app
from parsl.app.python import PythonApp
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.utils.apps import copy_data_future, pack

from .utils import (
    align_axes,
    app_filter,
    apply_offset,
    assign_identifiers,
    batch_frames,
    clean_frames,
    count_frames,
    extract_quantities,
    get_elements,
    get_train_valid_indices,
    insert_quantities,
    join_frames,
    not_null,
    read_frames,
    reset_frames,
    shuffle,
    write_frames,
)


@psiflow.register_serializable
class Dataset:
    """
    A class representing a dataset of atomic structures.

    This class provides methods for manipulating and analyzing collections of atomic structures.
    """

    extxyz: psiflow._DataFuture

    def __init__(
        self,
        states: Optional[list[AppFuture | Geometry] | AppFuture] = None,
        extxyz: Optional[psiflow._DataFuture] = None,
    ):
        """
        Initialize a Dataset.

        Args:
            states: List of Geometry instances or AppFutures representing geometries.
        """
        if extxyz is not None:  # takes precedence over states
            self.extxyz = extxyz
            return

        if not isinstance(states, list):  # AppFuture[list, geometry]
            states = [states]
        file = psiflow.context().new_file("data_", ".xyz")
        self.extxyz = write_frames(*states, outputs=[file]).outputs[0]

    def length(self) -> AppFuture:
        """
        Get the number of structures in the dataset.

        Returns:
            AppFuture: Future representing the number of structures.
        """
        return count_frames(self.extxyz)

    def shuffle(self) -> "Dataset":
        """
        Shuffle the order of structures in the dataset.
        """
        file = psiflow.context().new_file("data_", ".xyz")
        extxyz = shuffle(self.extxyz, outputs=[file]).outputs[0]
        return Dataset(extxyz=extxyz)

    def __getitem__(
        self, index: int | slice | list[int] | AppFuture
    ) -> Dataset | AppFuture:
        """
        Get a subset of the dataset or a single structure.

        Args:
            index: Integer, slice, list of integers, or AppFuture representing indices.

        Returns:
            Union[Dataset, AppFuture]: A new Dataset or an AppFuture of a single Geometry.
        """
        future_frames = read_frames(self.extxyz, indices=index)
        if isinstance(index, int):
            return future_frames[0]  # will return Geometry as Future

        # slice, list, AppFuture
        file = psiflow.context().new_file("data_", ".xyz")
        extxyz = write_frames(future_frames, outputs=[file]).outputs[0]
        return Dataset(extxyz=extxyz)

    def save(self, path: Path | str) -> DataFuture:
        """
        Save the dataset to a file.
        Returns:
            DataFuture: Future representing the file to which will be saved.
        """
        path = psiflow.resolve_and_check(Path(path))
        future = copy_data_future(inputs=[self.extxyz], outputs=[File(path)])
        return future.outputs[0]

    def geometries(self) -> AppFuture:
        """
        Get all geometries in the dataset.

        Returns:
            AppFuture: Future representing a list of Geometry instances.
        """
        return read_frames(self.extxyz)

    def __add__(self, dataset: Dataset) -> Dataset:
        """
        Concatenate two datasets.
        """
        file = psiflow.context().new_file("data_", ".xyz")
        future = join_frames(inputs=[self.extxyz, dataset.extxyz], outputs=[file])
        return Dataset(extxyz=future.outputs[0])

    def subtract_offset(self, **atomic_energies: float | AppFuture) -> Dataset:
        """
        Subtract atomic energy offsets from the dataset.
        """
        assert len(atomic_energies) > 0
        file = psiflow.context().new_file("data_", ".xyz")
        future = apply_offset(
            self.extxyz, subtract=True, **atomic_energies, outputs=[file]
        )
        return Dataset(extxyz=future.outputs[0])

    def add_offset(self, **atomic_energies) -> Dataset:
        """
        Add atomic energy offsets to the dataset.
        """
        assert len(atomic_energies) > 0
        file = psiflow.context().new_file("data_", ".xyz")
        future = apply_offset(
            self.extxyz, subtract=False, **atomic_energies, outputs=[file]
        )
        return Dataset(extxyz=future.outputs[0])

    def elements(self) -> AppFuture:
        """
        Get the set of elements present in the dataset.

        Returns:
            AppFuture: Future representing a set of element symbols.
        """
        return get_elements(self.extxyz)

    def reset(self) -> Dataset:
        """
        Reset all structures in the dataset.
        """
        file = psiflow.context().new_file("data_", ".xyz")
        future = reset_frames(self.extxyz, outputs=[file])
        return Dataset(extxyz=future.outputs[0])

    def clean(self) -> Dataset:
        """
        Clean all structures in the dataset.
        """

        file = psiflow.context().new_file("data_", ".xyz")
        future = clean_frames(self.extxyz, outputs=[file])
        return Dataset(extxyz=future.outputs[0])

    def get(
        self,
        *quantities: str,
        atom_indices: Optional[list[int]] = None,
        elements: Optional[list[str]] = None,
    ):
        """
        Extract specified quantities from the dataset.

        Args:
            *quantities: Names of quantities to extract.
            atom_indices: Optional list of atom indices to consider.
            elements: Optional list of element symbols to consider.

        Returns:
            Union[AppFuture, tuple[AppFuture, ...]]: Future(s) representing the extracted quantities.
        """
        result = extract_quantities(
            quantities,
            atom_indices,
            elements,
            inputs=[self.extxyz],
        )
        if len(quantities) == 1:
            return result[0]
        else:
            return tuple([result[i] for i in range(len(quantities))])

    def evaluate(
        self,
        computable: Computable,
        batch_size: Optional[int] = None,
    ) -> Dataset:
        """
        Evaluate a Computable on the dataset.

        Args:
            computable: Computable object to evaluate.
            batch_size: Optional batch size for evaluation.

        Returns:
            Dataset: A new Dataset with evaluation results.
        """
        # TODO: remove this functionality?
        from psiflow.hamiltonians import Hamiltonian

        if not isinstance(computable, Hamiltonian):
            # avoid extracting and inserting the same quantities
            return computable.compute_dataset(self)

        # use Hamiltonian.compute method
        if batch_size is not None:
            outputs = computable.compute(self, batch_size=batch_size)
        else:
            outputs = computable.compute(self)  # use default from computable
        if not isinstance(outputs, list):  # compute unpacks for only one property
            outputs = [outputs]
        future = insert_quantities(
            quantities=tuple(computable.outputs),
            arrays=pack(*outputs),
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        )
        return Dataset(None, future.outputs[0])

    def filter(
        self,
        quantity: str,
    ) -> Dataset:
        """
        Filter the dataset based on a specified quantity.

        Args:
            quantity: The quantity to filter on.

        Returns:
            Dataset: A new Dataset containing only structures that pass the filter.
        """
        assert quantity in QUANTITIES
        extxyz = app_filter(
            quantity,
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def not_null(self) -> Dataset:
        """
        Remove null states from the dataset.

        Returns:
            Dataset: A new Dataset without null states.
        """
        file = psiflow.context().new_file("data_", ".xyz")
        extxyz = not_null(inputs=[self.extxyz], outputs=[file]).outputs[0]
        return Dataset(None, extxyz)

    def align_axes(self):
        """
        Adopt a canonical orientation for all (periodic) structures in the dataset.

        Returns:
            Dataset: A new Dataset with aligned structures.
        """
        extxyz = align_axes(
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        ).outputs[0]
        return Dataset(None, extxyz)

    def split(self, fraction, shuffle=True):  # auto-shuffles
        """
        Split the dataset into training and validation sets.

        Args:
            fraction: Fraction of data to use for training.
            shuffle: Whether to shuffle before splitting.

        Returns:
            tuple[Dataset, Dataset]: Training and validation datasets.
        """
        train, valid = get_train_valid_indices(
            self.length(),
            fraction,
            shuffle,
        )
        return self.__getitem__(train), self.__getitem__(valid)

    def assign_identifiers(
        self, identifier: Union[int, AppFuture, None] = None
    ) -> AppFuture:
        """
        Assign identifiers to structures in the dataset.

        Args:
            identifier: Starting identifier or AppFuture representing it.

        Returns:
            AppFuture: Future representing the next available identifier.
        """
        result = assign_identifiers(
            identifier,
            inputs=[self.extxyz],
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        )
        self.extxyz = result.outputs[0]
        return result

    @classmethod
    def load(cls, path_xyz: Union[Path, str]) -> Dataset:
        """
        Load a dataset from a file.
        """
        path_xyz = psiflow.resolve_and_check(Path(path_xyz))
        assert path_xyz.exists()  # needs to be locally accessible
        return cls(extxyz=File(path_xyz))


def _concatenate_multiple(*args: list[np.ndarray]) -> list[np.ndarray]:
    """
    Concatenate multiple lists of arrays.

    Args:
        *args: Lists of numpy arrays to concatenate.

    Returns:
        list[np.ndarray]: List of concatenated arrays.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """

    def pad_arrays(
        arrays: list[np.ndarray],
        pad_dimension: int = 1,
    ) -> list[np.ndarray]:
        ndims = np.array([len(a.shape) for a in arrays])
        assert np.all(ndims == ndims[0])
        assert np.all(pad_dimension < ndims)

        pad_size = max([a.shape[pad_dimension] for a in arrays])
        for i in range(len(arrays)):
            shape = list(arrays[i].shape)
            shape[pad_dimension] = pad_size - shape[pad_dimension]
            padding = np.zeros(tuple(shape)) + np.nan
            arrays[i] = np.concatenate((arrays[i], padding), axis=pad_dimension)
        return arrays

    narrays = len(args[0])
    for arg in args:
        assert isinstance(arg, list)
    assert all([len(a) == narrays for a in args])

    concatenated = []
    for i in range(narrays):
        arrays = [arg[i] for arg in args]
        if len(arrays[0].shape) > 1:
            pad_arrays(arrays)
        concatenated.append(np.concatenate(tuple(arrays)))
    return concatenated


concatenate_multiple = python_app(_concatenate_multiple, executors=["default_threads"])


def _aggregate_multiple(
    *arrays_list,
    coefficients: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """
    Aggregate multiple lists of arrays with optional coefficients.

    Args:
        *arrays_list: Lists of arrays to aggregate.
        coefficients: Optional coefficients for weighted aggregation.

    Returns:
        list[np.ndarray]: List of aggregated arrays.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    if coefficients is None:
        coefficients = np.ones(len(arrays_list))
    else:
        assert len(coefficients) == len(arrays_list)

    results = [np.zeros(a.shape) for a in arrays_list[0]]
    for i, arrays in enumerate(arrays_list):
        for j, array in enumerate(arrays):
            results[j] += coefficients[i] * array
    return results


aggregate_multiple = python_app(_aggregate_multiple, executors=["default_threads"])


@join_app
def batch_apply(
    apply_apps: tuple[Union[PythonApp, Callable]],
    arg: Union[Dataset, list[Geometry]],
    batch_size: int,
    length: int,
    outputs: list = [],
    reduce_func: Optional[PythonApp] = None,
    **app_kwargs,
) -> AppFuture:
    """
    Apply a set of apps to batches of data.

    Args:
        apply_apps: Tuple of PythonApps or Callables to apply.
        arg: Dataset or list of Geometries to process.
        batch_size: Size of each batch.
        length: Total number of items to process.
        outputs: List of output files.
        reduce_func: Optional function to reduce results.
        **app_kwargs: Additional keyword arguments for the apps.

    Returns:
        AppFuture: Future representing the result of batch application.

    Note:
        This function is wrapped as a Parsl join_app.
    """
    nbatches = math.ceil(length / batch_size)
    batches = [psiflow.context().new_file("data_", ".xyz") for _ in range(nbatches)]
    future = batch_frames(batch_size, inputs=[arg.extxyz], outputs=batches)
    output_futures = []
    for i in range(nbatches):
        futures = []
        for app in apply_apps:
            f = app(
                None,
                inputs=[future.outputs[i]],
                **app_kwargs,
            )
            futures.append(f)
        reduced = reduce_func(*futures)
        output_futures.append(reduced)
    future = concatenate_multiple(*output_futures)
    return future


@python_app(executors=["default_threads"])
def get_length(arg):
    """
    Get the length of the input argument.

    Args:
        arg: Input to get the length of.

    Returns:
        int: Length of the input.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    if isinstance(arg, list):
        return len(arg)
    else:
        return 1


def compute(
    arg: Union[Dataset, AppFuture[list], list, AppFuture, Geometry],
    *apply_apps: Union[PythonApp, Callable],
    outputs_: Union[str, list[str], tuple[str, ...], None] = None,
    reduce_func: Union[PythonApp, Callable] = aggregate_multiple,
    batch_size: Optional[int] = None,
) -> Union[list[AppFuture], AppFuture]:
    """
    Compute results by applying apps to the input data.

    Args:
        arg: Input data to compute on.
        *apply_apps: Apps to apply to the data.
        outputs_: Names of output quantities.
        reduce_func: Function to reduce results.
        batch_size: Optional batch size for processing.

    Returns:
        Union[list[AppFuture], AppFuture]: Future(s) representing computation results.
    """
    if type(outputs_) is str:
        outputs_ = [outputs_]
    if batch_size is not None:
        if isinstance(arg, Dataset):
            length = arg.length()
        else:
            length = get_length(arg)
            # convert to Dataset for convenience
            arg = Dataset(arg)
        future = batch_apply(
            apply_apps,
            arg,
            batch_size,
            length,
            outputs_=outputs_,
            reduce_func=reduce_func,
        )
    else:
        futures = []
        if isinstance(arg, Dataset):
            for app in apply_apps:
                future = app(
                    None,
                    outputs_=outputs_,
                    inputs=[arg.extxyz],
                )
                futures.append(future)
        else:
            for app in apply_apps:
                future = app(
                    arg,
                    outputs_=outputs_,
                    inputs=[],
                )
                futures.append(future)
        future = reduce_func(*futures)
    if len(outputs_) == 1:
        return future[0]
    else:
        return [future[i] for i in range(len(outputs_))]


class Computable:
    """
    Base class for computable objects.

    Attributes:
        outputs (ClassVar[tuple[str, ...]]): Names of output quantities.
        batch_size (ClassVar[Optional[int]]): Default batch size for computation.
    """

    outputs: ClassVar[tuple[str, ...]] = ()
    batch_size: ClassVar[Optional[int]] = None

    def compute(
        self,
        arg: Union[Dataset, AppFuture[list], list, AppFuture, Geometry],
        *outputs: Optional[str],
        batch_size: Optional[int] = -1,  # if -1: take class default
    ) -> Union[list[AppFuture], AppFuture]:
        """
        Compute results for the given input.

        Args:
            arg: Input data to compute on.
            outputs: Names of output quantities.
            batch_size: Batch size for computation.

        Returns:
            Union[list[AppFuture], AppFuture]: Future(s) representing computation results.
        """
        raise NotImplementedError
