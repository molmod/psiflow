import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from flower.models.base import BaseModel
from flower.reference.base import BaseReference
from flower.sampling import Ensemble
from flower.data import Dataset
from flower.checks import Check


class LearningState:

    def __init__(
            self,
            model: BaseModel,
            reference: BaseReference,
            ensemble: Ensemble,
            train: Optional[Dataset] = None,
            validate: Optional[Dataset] = None,
            checks: Optional[List[Check]] = None,
            ):
        self.model = model
        self.reference = reference
        self.ensemble = ensemble
        self.train = train
        self.validate = validate
        self.checks = checks


class Algorithm:

    def __init__(self, path_output):
        self.path_output = Path(path_output)
        self.path_output.mkdir(parents=True, exist_ok=True)

    def dry_run(self, learning_state):
        raise NotImplementedError

    def run(self, learning_state):
        raise NotImplementedError

    def write(self, learning_state, iteration):
        path = self.path_output / str(iteration)
        path.mkdir(parents=False, exist_ok=True) # parent should exist

        path_config_raw = path / 'config.yaml'
        path_config = path / 'config_after_init.yaml'
        path_model = path / 'model_undeployed.pth'
        if learning_state.model.config_future is not None:
            learning_state.model.save(
                    path_config_raw,
                    path_config,
                    path_model,
                    )
        else:
            learning_state.model.save(path_config_raw)
        path_ensemble = path / 'ensemble'
        path_ensemble.mkdir(parents=False, exist_ok=True)
        learning_state.ensemble.save(path_ensemble)
        if learning_state.train is not None:
            learning_state.train.save(path / 'train.xyz')
        if learning_state.validate is not None:
            learning_state.validate.save(path / 'validate.xyz')


#def online_learning(
#        model,
#        reference,
#        ensemble,
#        train=None,
#        validate=None,
#        checks=[],
#        #niters=10,
#        #writer=None,
#        ):
#
#    return model, reference, ensemble, train, validate
