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


def _new_file(context):
    _, name = tempfile.mkstemp(
            suffix='.txt',
            prefix='new_',
            dir=context.path,
            )
    return name


def try_manual_plumed_linking():
    if 'PLUMED_KERNEL' not in os.environ.keys():
        # try linking manually
        if 'CONDA_PREFIX' in os.environ.keys(): # for conda environments
            p = 'CONDA_PREFIX'
        elif 'PREFIX' in os.environ.keys(): # for pip environments
            p = 'PREFIX'
        else:
            print('failed to set plumed .so kernel')
            pass
        path = os.environ[p] + '/lib/libplumedKernel.so'
        if os.path.exists(path):
            os.environ['PLUMED_KERNEL'] = path
            print('plumed kernel manually set at at : {}'.format(path))


def set_path_hills_plumed(plumed_input, path_hills):
    lines = plumed_input.split('\n')
    for i, line in enumerate(lines):
        if 'METAD' in line.split():
            line_before = line.split('FILE=')[0]
            line_after  = line.split('FILE=')[1].split()[1:]
            lines[i] = line_before + 'FILE={} '.format(path_hills) + ' '.join(line_after)
    return '\n'.join(lines)


def get_bias_plumed(plumed_input):
    allowed_keywords = ['METAD', 'RESTRAINT']
    found = False
    for key in allowed_keywords:
        lines = plumed_input.split('\n')
        for i, line in enumerate(lines):
            if key in line.split():
                assert not found
                cv = line.split('ARG=')[1].split()[0]
                bias = (key, cv)
                found = True
    return bias


def copy_file(inputs=[], outputs=[]):
    import shutil
    shutil.copyfile(inputs[0], outputs[0])
