import numpy as np

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT


def generate_dummy_data(natoms, nstates, number):
    atoms = Atoms(
            numbers=number * np.ones(natoms),
            positions=np.random.uniform(0, 1, size=(natoms, 3)),
            )
    atoms_list = []
    for i in range(nstates):
        _atoms = atoms.copy()
        _atoms.info['energy'] = np.random.uniform(0, 1)
        _atoms.arrays['forces'] = np.random.uniform(0, 1, size=(natoms, 3))
        _atoms.set_positions(np.random.uniform(0, 1, size=(natoms, 3)))
        atoms_list.append(_atoms)
    return atoms_list


def generate_emt_cu_data(a, nstates):
    atoms = bulk('Cu', 'fcc', a=a, cubic=True)
    atoms.calc = EMT()
    pos = atoms.get_positions()
    box = atoms.get_cell()
    atoms_list = []
    for i in range(nstates):
        atoms.set_positions(pos + np.random.uniform(-0.05, 0.05, size=(len(atoms), 3)))
        atoms.set_cell(box + np.random.uniform(-0.05, 0.05, size=(3, 3)))
        _atoms = atoms.copy()
        _atoms.calc = None
        _atoms.info['energy']   = atoms.get_potential_energy()
        _atoms.arrays['forces'] = atoms.get_forces()
        _atoms.info['stress']   = atoms.get_stress(voigt=False)
        atoms_list.append(_atoms)
    return atoms_list
