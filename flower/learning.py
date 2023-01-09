from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Tuple
import typeguard
from dataclasses import dataclass
from typing import Optional
import logging

from parsl.app.app import python_app

from flower.utils import get_train_valid_indices
from flower.data import Dataset
from flower.manager import Manager
from flower.models import BaseModel
from flower.reference import BaseReference
from flower.sampling import RandomWalker, PlumedBias
from flower.ensemble import Ensemble


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


@typeguard.typechecked
class BaseLearning:
    parameter_cls = None

    def __init__(self, **kwargs):
        self.parameters = self.__class__.parameter_cls(**kwargs)


@typeguard.typechecked
@dataclass
class RandomLearningParameters:
    nstates           : int = 20
    train_valid_split : int = 0.9


@typeguard.typechecked
class RandomLearning(BaseLearning):
    parameter_cls = RandomLearningParameters

    def run(
            self,
            manager: Manager,
            model: BaseModel,
            reference: BaseReference,
            walker: RandomWalker,
            checks: Optional[list] = None,
            bias: Optional[PlumedBias] = None,
            ) -> Tuple[Dataset, Dataset]:
        ensemble = Ensemble.from_walker(walker, nwalkers=self.parameters.nstates)
        ensemble.log()
        data = ensemble.sample(self.parameters.nstates, checks=checks)
        data = reference.evaluate(data)
        data_success = data.get(indices=data.success)
        train, valid = get_train_valid_indices(
                data_success.length(), # can be less than nstates
                self.parameters.train_valid_split,
                )
        data_train = data_success.get(indices=train)
        data_valid = data_success.get(indices=valid)
        data_train.log('data_train')
        data_valid.log('data_valid')
        model.initialize(data_train)
        epochs = model.train(data_train, data_valid)
        logger.info('trained model for {} epochs'.format(epochs.result()))
        manager.save(
                name='random',
                model=model,
                ensemble=ensemble,
                data_train=data_train,
                data_valid=data_valid,
                data_failed=data.get(indices=data.failed),
                )
        log = manager.log_wandb( # log training
                run_name='random_after',
                model=model,
                ensemble=ensemble,
                data_train=data_train,
                data_valid=data_valid,
                bias=bias,
                )
        log.result() # necessary to force execution of logging app
        return data_train, data_valid


@typeguard.typechecked
@dataclass
class OnlineLearningParameters(RandomLearningParameters):
    niterations : int = 10
    retrain_model_per_iteration : bool = True


@typeguard.typechecked
class OnlineLearning(BaseLearning):
    parameter_cls = OnlineLearningParameters

    def run(
            self,
            manager: Manager,
            model: BaseModel,
            reference: BaseReference,
            ensemble: Ensemble,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list] = None,
            ) -> Tuple[Dataset, Dataset]:
        if data_train is None:
            data_train = Dataset(model.context, [])
        if data_valid is None:
            data_valid = Dataset(model.context, [])
        for i in range(self.parameters.niterations):
            model.deploy()
            dataset = ensemble.sample(
                    self.parameters.nstates,
                    model=model,
                    checks=checks,
                    )
            data = reference.evaluate(dataset)
            data_success = data.get(indices=data.success)
            train, valid = get_train_valid_indices(
                    data_success.length(), # can be less than nstates
                    self.parameters.train_valid_split,
                    )
            data_train.append(data_success.get(indices=train))
            data_valid.append(data_success.get(indices=valid))
            if self.parameters.retrain_model_per_iteration:
                model.reset()
                model.initialize(data_train)
            data_train.log('data_train')
            data_valid.log('data_valid')
            epochs = model.train(data_train, data_valid)
            logger.info('trained model for {} epochs'.format(epochs.result()))
            manager.save(
                    name=str(i),
                    model=model,
                    ensemble=ensemble,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    )
            log = manager.log_wandb( # log training
                    run_name=str(i),
                    model=model,
                    ensemble=ensemble,
                    data_train=data_train,
                    data_valid=data_valid,
                    bias=ensemble.biases[0], # possibly None
                    )
            log.result() # necessary to force execution of logging app
        return data_train, data_valid
