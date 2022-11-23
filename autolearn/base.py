import covalent as ct
import torch
import numpy as np
from dataclasses import dataclass


def get_evaluate_dataset_electron(model_execution):
    device = model_execution.device
    ncores = model_execution.ncores
    dtype  = model_execution.dtype
    def evaluate_dataset_barebones(dataset, model):
        if device == 'cpu':
            torch.set_num_threads(ncores)
        if len(dataset) > 0:
            atoms = dataset.atoms_list[0].copy()
            atoms.calc = model.get_calculator(device, dtype)
            for state in dataset.atoms_list:
                atoms.set_positions(state.get_positions())
                atoms.set_cell(state.get_cell())
                state.info['energy']   = atoms.get_potential_energy()
                state.arrays['forces'] = atoms.get_forces()
                try: # some models do not have stress support
                    state.info['stress'] = atoms.get_stress(voigt=False)
                except Exception as e:
                    print(e)
                    state.info['stress'] = np.zeros((3, 3))
        return dataset
    return ct.electron(evaluate_dataset_barebones, executor=model_execution.executor)


class BaseModel:
    """Base class for a trainable interaction potential"""

    def get_calculator(self, device, dtype):
        raise NotImplementedError

    @staticmethod
    def train(model, dataset, training_execution):
        """Trains a model and returns it as a covalent electron"""
        raise NotImplementedError

    @staticmethod
    def evaluate(dataset, model, model_execution):
        """Evaluates a dataset using a model and returns it as a covalent electron"""
        evaluate_electron = get_evaluate_dataset_electron(model_execution)
        return evaluate_electron(dataset, model)


class BaseReference:
    """Base class for a reference interaction potential"""

    def get_calculator(self):
        raise NotImplementedError

    @staticmethod
    def evaluate(atoms, reference, reference_execution):
        """Evaluates an atoms configuration and returns it as a covalent electron"""
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
    executor: str = 'local'
    device  : str = 'cpu'
    ncores  : int = 1
