from typing import Optional, Callable
from dataclasses import dataclass
import torch
import numpy as np

from ase.io import write

from autolearn import Dataset
from autolearn.execution import ModelExecutionDefinition, \
        TrainingExecutionDefinition


class BaseReference:
    """Base class for a reference interaction potential"""

    def evaluate(self, sample, reference_execution):
        """Evaluates and labels a sample and returns it as a covalent electron"""
        raise NotImplementedError

    def evaluate_dataset(self, dataset, reference_execution):
        return Dataset([self.evaluate(s, reference_execution) for s in dataset])


class BaseWalker:

    def propagate(self, model, model_execution):
        raise NotImplementedError

    def reset(self):
        self.state = deepcopy(self.start)
        return self

    def sample(self):
        return self.state
