from dataclasses import dataclass
from copy import deepcopy

from parsl.app.app import python_app
from parsl.data_provider.files import File

from autolearn.dataset import save_dataset
from autolearn.execution import ModelExecutionDefinition


def copy_atoms(atoms):
    from ase import Atoms
    _atoms = Atoms(
            numbers=atoms.numbers.copy(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=True,
            )
    return _atoms


@dataclass
class EmptyParameters:
    pass


class BaseWalker:
    parameters_cls = EmptyParameters

    def __init__(self, context, start, **kwargs):
        self.context = context
        self.tag    = 'reset'
        self.p_copy_atoms = python_app(
                copy_atoms,
                executors=[self.executor_label],
                )
        self.start = self.p_copy_atoms(start)
        self.state = self.p_copy_atoms(start)
        self.parameters = self.parameters_cls(**kwargs)

    def propagate(self, model):
        raise NotImplementedError

    def reset(self):
        self.state = self.p_copy_atoms(self.start)
        self.tag   = 'reset'

    def save(self, path_xyz):
        p_copy_file = python_app(save_dataset, executors=[self.executor_label])
        return p_copy_file(self.state)

    def copy(self):
        walker = self.__class__(
                self.context,
                self.start,
                )
        walker.parameters = deepcopy(self.parameters)
        return walker

    @property
    def is_safe(self):
        def _is_safe(tag):
            if (tag == 'safe') or (tag == 'reset'):
                return True
            return False
        p_is_safe = python_app(
                _is_safe,
                executors=[self.executor_label],
                )
        return p_is_safe(self.tag)

    @property
    def executor_label(self):
        return self.context[ModelExecutionDefinition].executor_label
