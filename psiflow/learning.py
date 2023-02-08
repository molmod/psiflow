from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Tuple
import typeguard
from dataclasses import dataclass
from typing import Optional
import logging

from parsl.dataflow.futures import AppFuture

from psiflow.utils import get_train_valid_indices
from psiflow.data import Dataset
from psiflow.experiment import FlowManager
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.sampling import RandomWalker, PlumedBias
from psiflow.ensemble import Ensemble


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


@typeguard.typechecked
@dataclass
class BaseLearning:
    train_valid_split: float = 0.9
    pretraining_nstates: int = 50
    pretraining_amplitude_pos: float = 0.05
    pretraining_amplitude_box: float = 0.05

    def run_pretraining(
            self,
            flow_manager: FlowManager,
            model: BaseModel,
            reference: BaseReference,
            initial_data: Dataset,
            ):
        nstates = self.pretraining_nstates
        amplitude_pos = self.pretraining_amplitude_pos
        amplitude_box = self.pretraining_amplitude_box
        logger.info('performing random pretraining')
        walker = RandomWalker( # create walker for state i in initial data
                reference.context,
                initial_data[0],
                amplitude_pos=amplitude_pos,
                amplitude_box=amplitude_box,
                )
        ensemble = Ensemble.from_walker(
                walker,
                self.pretraining_nstates,
                dataset=initial_data,
                )
        random_data = ensemble.sample(self.pretraining_nstates)
        data = reference.evaluate(random_data)
        data_success = data.get(indices=data.success)
        train, valid = get_train_valid_indices(
                data_success.length(), # can be less than nstates
                self.train_valid_split,
                )
        data_train = data_success.get(indices=train)
        data_valid = data_success.get(indices=valid)
        data_train.log('data_train')
        data_valid.log('data_valid')
        model.initialize(data_train)
        model.train(data_train, data_valid)
        flow_manager.save(
                name='random_pretraining',
                model=model,
                ensemble=ensemble,
                data_train=data_train,
                data_valid=data_valid,
                data_failed=data.get(indices=data.failed),
                require_done=False,
                )
        log = flow_manager.log_wandb( # log training
                run_name='random_pretraining',
                model=model,
                ensemble=ensemble,
                data_train=data_train,
                data_valid=data_valid,
                )
        return data_train, data_valid


@typeguard.typechecked
@dataclass
class BatchLearning(BaseLearning):
    nstates: int = 30
    niterations: int = 10
    retrain_model_per_iteration : bool = True

    def run(
            self,
            flow_manager: FlowManager,
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
        for i in range(self.niterations):
            if flow_manager.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            model.deploy()
            dataset = ensemble.sample(
                    self.nstates,
                    model=model,
                    checks=checks,
                    )
            data = reference.evaluate(dataset)
            data_success = data.get(indices=data.success)
            train, valid = get_train_valid_indices(
                    data_success.length(), # can be less than nstates
                    self.train_valid_split,
                    )
            data_train.append(data_success.get(indices=train))
            data_valid.append(data_success.get(indices=valid))
            if self.retrain_model_per_iteration:
                logger.info('reinitialize model (scale/shift/avg_num_neighbors) on new training data')
                model.reset()
                model.initialize(data_train)
            data_train.log('data_train')
            data_valid.log('data_valid')
            model.train(data_train, data_valid)
            flow_manager.save(
                    name=str(i),
                    model=model,
                    ensemble=ensemble,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    )
            log = flow_manager.log_wandb( # log training
                    run_name=str(i),
                    model=model,
                    ensemble=ensemble,
                    data_train=data_train,
                    data_valid=data_valid,
                    bias=ensemble.biases[0], # possibly None
                    )
            log.result() # necessary to force execution of logging app
        return data_train, data_valid


@typeguard.typechecked
def train_online(
        states: list[AppFuture],
        models: list[BaseModel],
        data_train: Dataset,
        data_valid: Dataset,
        retrain_threshold: int,
        train_valid_split: float,
        ) -> None:
    nfinished = sum([state.done() for state in states])
    logger.info('found {} finished states'.format(nfinished))
    if nfinished >= retrain_threshold:
        if not models[-1].model_future.done():
            return None # do not restart training if still busy
        done = []
        i = 0
        while i < len(states):
            if states[i].done():
                done.append(states[i])
                states.pop(i)
            else:
                i += 1
        assert len(done) >= retrain_threshold
        data = Dataset(models[0].context, done)
        data_success = data.get(indices=data.success)
        train, valid = get_train_valid_indices(
                data_success.length(), # can be less than nstates
                train_valid_split,
                )
        data_train.append(data_success.get(indices=train))
        data_valid.append(data_success.get(indices=valid))
        data_train.log('data_train')
        data_valid.log('data_valid')
        model_ = model.copy()
        models.append(model_)
        model_.train(data_train, data_valid)
    else:
        pass


@typeguard.typechecked
@dataclass
class OnlineLearning(BaseLearning):
    retrain_threshold: int = 20

    def run(
            self,
            flow_manager: FlowManager,
            model: BaseModel,
            reference: BaseReference,
            ensemble: Ensemble,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list] = None,
            ) -> Tuple[Dataset, Dataset]:
        iterator = zip(ensemble.walkers, ensemble.biases)
        models = [model]
        counter = 1 # counts number of models which are 'ready'
        flow_manager.save(
                name='model_{}'.format(counter),
                model=models[0],
                ensemble=ensemble,
                data_train=data_train,
                data_valid=data_valid,
                )
        while True:
            states = []
            for i, (walker, bias) in enumerate(iterator):
                model = None
                for m in models[::-1]: # select newest model which is ready
                    if m.model_future.done():
                        model = m
                        if len(model.deploy_future) == 0:
                            model.deploy()
                        break
                if model is None:
                    logger.info('did not find a model that was ready for walker propagation')
                    logger.info('the following models exist: ')
                    for m in models[::-1]:
                        logger.info(m)
                        logger.info(m.model_future.filepath)
                    continue # no model ready
                if walker.state_future.done():
                    logger.info('walker {} is evaluated; continuing propagation'.format(i))
                    states.append(reference.evaluate(walker.state_future))
                    walker.reset_if_unsafe()
                    walker.propagate(model=model, bias=bias)
                else:
                    pass
            train_online(
                    states,
                    models,
                    data_train,
                    data_valid,
                    self.retrain_threshold,
                    self.train_valid_split,
                    )
            if len(models) > counter: # new model was added, save new data
                counter += 1
                flow_manager.save(
                        name='model_{}'.format(counter),
                        model=models[-1], # save model and its training data
                        data_train=data_train,
                        data_valid=data_valid,
                        require_done=False, # only completes when model is trained
                        )
