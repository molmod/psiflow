import covalent as ct

from ase.data import chemical_symbols


def get_numbers(atoms_list):
    _all = [set(a.numbers) for a in atoms_list]
    return sorted(list(set(b for a in _all for b in a)))


def get_elements(atoms_list):
    numbers = get_numbers(atoms_list)
    return [chemical_symbols[n] for n in numbers]


@ct.electron(executor='local')
def prepare_dict(my_dict):
    return dict(my_dict)
