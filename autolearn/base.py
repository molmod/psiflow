from typing import Optional, Callable
from dataclasses import dataclass
import covalent as ct
import torch
import numpy as np

from ase.io import write


class BaseModel:
    """Base class for a trainable interaction potential"""

    def get_calculator(self, device, dtype):
        raise NotImplementedError

    def train(self, dataset, training_execution):
        """Trains a model and returns it as a covalent electron"""
        raise NotImplementedError

    def evaluate(self, dataset, model_execution):
        """Evaluates a dataset using a model and returns it as a covalent electron"""
        device = model_execution.device
        ncores = model_execution.ncores
        dtype  = model_execution.dtype
        def evaluate_dataset_barebones(dataset, model):
            if device == 'cpu':
                torch.set_num_threads(ncores)
            if len(dataset) > 0:
                atoms = dataset.as_atoms_list()[0].copy()
                atoms.calc = model.get_calculator(device, dtype)
                for sample in dataset.samples:
                    atoms.set_positions(sample.atoms.get_positions())
                    atoms.set_cell(sample.atoms.get_cell())
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    try: # some models do not have stress support
                        stress = atoms.get_stress(voigt=False)
                    except Exception as e:
                        print(e)
                        stress = np.zeros((3, 3))
                    sample.label(energy, forces, stress, log=None)
            return dataset
        evaluate_electron = ct.electron(
                evaluate_dataset_barebones,
                executor=model_execution.executor,
                )
        return evaluate_electron(dataset, self)


class BaseReference:
    """Base class for a reference interaction potential"""

    def evaluate(self, sample, reference_execution):
        """Evaluates and labels a sample and returns it as a covalent electron"""
        raise NotImplementedError


class BaseWalker:

    def proceed(self, model, model_execution):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def sample(self, model):
        raise NotImplementedError


@dataclass(frozen=True)
class TrainingExecution:
    executor: str = 'local'
    device  : str = 'cuda'


@dataclass(frozen=True)
class ModelExecution:
    executor: str = 'local'
    device  : str = 'cpu'
    ncores  : int = 1
    dtype   : str = 'float32'


@dataclass(frozen=True)
class ReferenceExecution:
    executor : str  = 'local'
    ncores   : int  = 1
    command  : str  = 'cp2k.psmp' # default command for CP2K Reference
    mpi      : Optional[Callable] = None # or callable, e.g: mpi(ncores) -> 'mpirun -np {ncores} '
    walltime : int  = 3600 # timeout in seconds
