import os
import tempfile
import numpy as np

from ase.data import chemical_symbols, atomic_numbers

from parsl.app.app import python_app


def get_index_element_mask(numbers, elements, atom_indices):
    mask = np.array([True] * len(numbers))

    if elements is not None:
        numbers_to_include = [atomic_numbers[e] for e in elements]
        mask_elements = np.array([False] * len(numbers))
        for number in numbers_to_include:
            mask_elements = np.logical_or(mask_elements, (numbers == number))
        mask = np.logical_and(mask, mask_elements)

    if atom_indices is not None:
        mask_indices = np.array([False] * len(numbers))
        mask_indices[np.array(atom_indices)] = True
        mask = np.logical_and(mask, mask_indices)
    return mask


def _new_file(path, prefix, suffix):
    _, name = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=path,
            )
    return name


@python_app
def copy_data_future(inputs=[], outputs=[]):
    import shutil
    shutil.copyfile(inputs[0], outputs[0])


@python_app
def copy_app_future(future):
    from copy import deepcopy
    return deepcopy(future)


@python_app
def unpack_i(result, i):
    return result[i]


@python_app
def save_yaml(input_dict, outputs=[]):
    import yaml
    with open(outputs[0], 'w') as f:
        yaml.dump(input_dict, f, default_flow_style=False)


@python_app
def save_atoms(atoms, outputs=[]):
    from ase.io import write
    write(outputs[0].filepath, atoms)


@python_app
def save_txt(data, outputs=[]):
    with open(outputs[0], 'w') as f:
        f.write(data)
