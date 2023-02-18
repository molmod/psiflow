import numpy as np

from ase.data import atomic_numbers

from psiflow.data import FlowAtoms, Dataset


def generate_isolated_atoms(elements, box_size):
    atoms_list = []
    for element in elements:
        atoms = FlowAtoms(
                numbers=np.array([atomic_numbers[element]]),
                positions=np.array([0, 0, 0]),
                cell=np.eye(3) * box_size,
                pbc=True,
                )
        atoms_list.append(atoms)
    return Dataset(atoms_list)
