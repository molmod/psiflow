import numpy as np
import covalent as ct

@ct.electron(executor='local')
def prepare_dict(my_dict):
    return dict(my_dict)


def clear_label(atoms):
    atoms.info['energy'] = 0.0
    atoms.arrays['forces'] = np.zeros((len(atoms), 3))
    if 'stress' in atoms.info.keys():
        atoms.info['stress'] = np.zeros((3, 3))
