import covalent as ct
import numpy as np


class Sample:
    """Wrapper class for a sample in phase space"""
    allowed_tags = [
            'success',
            'error',
            'timeout',
            'unsafe',
            ]

    def __init__(self, atoms):
        _atoms = atoms.copy()
        _atoms.calc = None

        self.evaluated = (('energy' in _atoms.info.keys()) and
                ('stress' in _atoms.info.keys()) and
                ('forces' in _atoms.arrays.keys()))
        self.tags  = _atoms.info.pop('tags', [])
        self.atoms = _atoms
        self.log   = None

    def label(self, energy, forces, stress, log=None):
        assert isinstance(energy, float)
        assert forces.shape == (len(self.atoms), 3)
        assert stress.shape == (3, 3)
        self.evaluated = True
        self.atoms.info['energy'] = energy
        self.atoms.info['stress'] = stress
        self.atoms.arrays['forces'] = forces
        self.log = log

    def tag(self, tag):
        assert tag in Sample.allowed_tags
        self.tags.append(tag)

    def get_atoms():
        _atoms = self.atoms.copy()
        _atoms.info['tags'] = ' '.join(self.tags)
        return _atoms

    def clear(self):
        self.tags = []
        self.label(0.0, np.zeros((len(self.atoms), 3)), np.zeros((3, 3)))
        self.evaluated = False


class Dataset:
    """Base class for a dataset of atomic structures"""

    def __init__(self, samples=[]):
        """Constructor

        Arguments
        ---------

        samples : list of Sample objects

        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for sample in self.samples:
            yield sample

    def __getitem__(self, key):
        return self.samples[key]

    def as_atoms_list(self):
        return [sample.atoms for sample in self.samples]

    def clear(self):
        for sample in self.samples:
            sample.clear()

    @classmethod
    def from_atoms_list(cls, atoms_list):
        return cls([Sample(atoms) for atoms in atoms_list])

    #@staticmethod
    #@ct.electron(executor='local')
    #def clear_labels(dataset):
    #    for atoms in dataset.atoms_list:
    #        clear_label(atoms)
    #    return dataset
