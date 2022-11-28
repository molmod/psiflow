import os
import covalent as ct

from ase.data import chemical_symbols


def get_numbers(atoms_list):
    _all = [set(a.numbers) for a in atoms_list]
    return sorted(list(set(b for a in _all for b in a)))


def get_elements(atoms_list):
    numbers = get_numbers(atoms_list)
    return [chemical_symbols[n] for n in numbers]


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
