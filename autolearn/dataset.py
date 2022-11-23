import covalent as ct
import numpy as np

from ase.data import chemical_symbols

from autolearn.utils import clear_label


class Dataset:
    """Base class for a dataset of atomic structures"""

    def __init__(self, atoms_list=[]):
        """Constructor

        Arguments
        ---------

        atoms_list : list of Atoms
            list of atoms instances

        """
        # remove calc attribute to avoid confusion or pickling problems
        for atoms in atoms_list:
            atoms.calc = None
        self.atoms_list = atoms_list

    def get_elements(self):
        numbers = self.get_numbers()
        return [chemical_symbols[n] for n in numbers]

    def get_numbers(self):
        """Returns set of all atomic numbers that are present in the data"""
        _all = [set(a.numbers) for a in self.atoms_list]
        return sorted(list(set(b for a in _all for b in a)))

    def __len__(self):
        return len(self.atoms_list)

    @staticmethod
    @ct.electron(executor='local')
    def clear_labels(dataset):
        for atoms in dataset.atoms_list:
            clear_label(atoms)
        return dataset

    @staticmethod
    def label(dataset):
        pass
