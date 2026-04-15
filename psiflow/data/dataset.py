from pathlib import Path
from typing import Optional, Union

from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.utils.apps import copy_data_future

from psiflow.data.file import (
    count_frames,
    get_elements,
    join_frames,
    read_frames,
    write_frames,
    split_frames,
    shuffle_frames,
    apply_offset,
    reset_frames,
    clean_frames,
    align_axes,
    filter_frames,
    extract_quantities,
    extract_quantities_per_atom,
    assign_identifiers,
)

# TODO: do all operations need to generate a new file?


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
        Initialize a Dataset. Provide either states or extxyz.

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
        extxyz = shuffle_frames(self.extxyz, outputs=[file]).outputs[0]
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

    def get(self, *quantities: str) -> tuple[AppFuture, ...]:
        """
        Extract specified quantities from the dataset.
        """
        future_dict = extract_quantities(self.extxyz, quantities)
        return tuple(future_dict[q] for q in quantities)

    def get_per_atom(
        self,
        *quantities: str,
        atom_indices: Optional[list[int]] = None,
        elements: Optional[list[str]] = None,
    ) -> tuple[AppFuture, ...]:
        """
        Extract specified per atom quantities from the dataset.

        Args:
            *quantities: Names of quantities to extract.
            atom_indices: Optional list of atom indices to consider.
            elements: Optional list of element symbols to consider.
        """
        future_dict = extract_quantities_per_atom(
            self.extxyz, quantities, atom_indices, elements
        )
        return tuple(future_dict[q] for q in quantities)

    # TODO: cleanup?
    # def evaluate(
    #     self, computable: "Computable", batch_size: Optional[int] = None
    # ) -> Dataset:
    #     """
    #     Evaluate a Computable on the dataset.
    #     """
    #     # TODO: remove this functionality?
    #     #  or this should be the (only) way to label a dataset?
    #     from psiflow.hamiltonians import Hamiltonian
    #
    #     if not isinstance(computable, Hamiltonian):
    #         # avoid extracting and inserting the same quantities
    #         return computable.compute_dataset(self)
    #
    #     # use Hamiltonian.compute method
    #     if batch_size is not None:
    #         outputs = computable.compute(self, batch_size=batch_size)
    #     else:
    #         outputs = computable.compute(self)  # use default from computable
    #     if not isinstance(outputs, list):  # compute unpacks for only one property
    #         outputs = [outputs]
    #     data = {k: v for k, v in zip(computable.outputs, outputs)}
    #
    #     file = psiflow.context().new_file("data_", ".xyz")
    #
    #     future = read_frames(self.extxyz)
    #     future = insert_quantities(future, data)
    #     extxyz = write_frames(future, outputs=[file]).outputs[0]
    #     return Dataset(extxyz=extxyz)

    def filter(self, quantity: str) -> Dataset:
        """
        Filter the dataset based on a specified quantity.
        """
        # TODO: where is this used?
        file = psiflow.context().new_file("data_", ".xyz")
        future = filter_frames(self.extxyz, quantity, outputs=[file])
        return Dataset(extxyz=future.outputs[0])

    def align_axes(self) -> Dataset:
        """
        Adopt a canonical orientation for all (periodic) structures in the dataset.
        """
        file = psiflow.context().new_file("data_", ".xyz")
        future = align_axes(self.extxyz, outputs=[file])
        return Dataset(extxyz=future.outputs[0])

    def split(self, fraction: float, shuffle: bool = True) -> tuple[Dataset, Dataset]:
        """
        Split the dataset into training and validation sets.

        Args:
            fraction: Fraction of data to use for training.
            shuffle: Whether to shuffle before splitting.
        """
        assert 0 <= fraction <= 1
        file_train = psiflow.context().new_file("data_", ".xyz")
        file_val = psiflow.context().new_file("data_", ".xyz")
        future = split_frames(
            self.extxyz, fraction, shuffle, outputs=[file_train, file_val]
        )
        return Dataset(extxyz=future.outputs[0]), Dataset(extxyz=future.outputs[1])

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
        # TODO: what is the use case for this?
        # TODO: this is the only method that changes inplace instead of returning a new Dataset
        future = assign_identifiers(self.extxyz, identifier)
        future_states, future_id = future[0], future[1]
        file = psiflow.context().new_file("data_", ".xyz")
        self.extxyz = write_frames(future_states, outputs=[file]).outputs[0]
        return future_id

    @classmethod
    def load(cls, path_xyz: Union[Path, str]) -> Dataset:
        """
        Load a dataset from a file.
        """
        path_xyz = psiflow.resolve_and_check(Path(path_xyz))
        assert path_xyz.exists()  # needs to be locally accessible
        return cls(extxyz=File(path_xyz))
