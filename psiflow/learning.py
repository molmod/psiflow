from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import yaml
import logging

from concurrent.futures import as_completed
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.utils import get_train_valid_indices, save_yaml
from psiflow.data import Dataset
from psiflow.wandb_utils import WandBLogger
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.sampling import RandomWalker, PlumedBias
from psiflow.generator import Generator
from psiflow.checks import Check
from psiflow.state import save_state, load_state


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
@dataclass
class BaseLearning:
    path_output: Union[Path, str]
    train_valid_split: float = 0.9
    pretraining_nstates: int = 50
    pretraining_amplitude_pos: float = 0.05
    pretraining_amplitude_box: float = 0.05
    wandb_logger: Optional[WandBLogger] = None

    def __post_init__(self) -> None: # save self in output folder
        self.path_output = Path(self.path_output)
        if not self.path_output.is_dir():
            self.path_output.mkdir()
        config = asdict(self)
        config['path_output'] = str(self.path_output) # yaml requires str
        config.pop('wandb_logger')
        if self.wandb_logger is not None:
            config['WandBLogger'] = self.wandb_logger.parameters()
        path_config = self.path_output / (self.__class__.__name__ + '.yaml')
        if path_config.is_file():
            logger.warning('overriding learning config file {}'.format(path_config))
        save_yaml(config, outputs=[File(str(path_config))]).result()

    def output_exists(self, name):
        return (self.path_output / name).is_dir()

    def run_pretraining(
            self,
            model: BaseModel,
            reference: BaseReference,
            initial_data: Dataset,
            ):
        nstates = self.pretraining_nstates
        amplitude_pos = self.pretraining_amplitude_pos
        amplitude_box = self.pretraining_amplitude_box
        logger.info('performing random pretraining')
        walker = RandomWalker( # create walker for state i in initial data
                initial_data[0],
                amplitude_pos=amplitude_pos,
                amplitude_box=amplitude_box,
                )
        generators = Generator('random', walker, None).multiply(
                self.pretraining_nstates,
                initialize_using=initial_data,
                )
        states = [generator(None, reference) for generator in generators]
        data = Dataset(states)
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
        model.deploy()
        save_state(
                self.path_output,
                name='random_pretraining',
                model=model,
                generators=generators,
                data_train=data_train,
                data_valid=data_valid,
                data_failed=data.get(indices=data.failed),
                require_done=False,
                )
        if self.wandb_logger is not None:
            log = self.wandb_logger( # log training
                    run_name='random_pretraining',
                    model=model,
                    generators=generators,
                    data_train=data_train,
                    data_valid=data_valid,
                    )
            log.result() # force execution
        return data_train, data_valid


@typeguard.typechecked
@dataclass
class SequentialLearning(BaseLearning):
    nstates: int = 30
    niterations: int = 10
    retrain_model_per_iteration : bool = True

    def run(
            self,
            model: BaseModel,
            reference: BaseReference,
            generators: list[Generator],
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list[Check]] = None,
            ) -> tuple[Dataset, Dataset]:
        if self.wandb_logger is not None:
            self.wandb_logger.insert_name(model)
        if data_train is None:
            data_train = Dataset([])
        if data_valid is None:
            data_valid = Dataset([])
        assert model.model_future is not None, ('model has to be initialized '
                'before running batch learning')
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            model.deploy()
            states = [g(model, reference, checks) for g in generators]
            data = Dataset(states)
            data_success = data.get(indices=data.success)
            train, valid = get_train_valid_indices(
                    data_success.length(),
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
            self.flow_manager.save(
                    name=str(i),
                    model=model,
                    generators=generators,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    require_done=False,
                    )
            if self.wandb_logger is not None:
                log = self.wandb_logger( # log training
                        run_name=str(i),
                        model=model,
                        generators=generators,
                        data_train=data_train,
                        data_valid=data_valid,
                        bias=generators[0].bias, # possibly None
                        )
                log.result() # necessary to force execution of logging app
        return data_train, data_valid


@typeguard.typechecked
@dataclass
class ConcurrentLearning(BaseLearning):
    niterations: int = 10
    retrain_model_per_iteration : bool = True
    retrain_threshold: int = 20

    def run(
            self,
            model: BaseModel,
            reference: BaseReference,
            generators: list[Generator],
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list] = None,
            ) -> tuple[Dataset, Dataset]:
        if self.wandb_logger is not None:
            self.wandb_logger.insert_name(model)
        @join_app
        def delayed_deploy(model, wait_for_it):
            model.deploy()

        assert model.model_future is not None, ('model has to be initialized '
                'before running online learning')
        ngenerators = len(generators)
        assert self.retrain_threshold < ngenerators, ('the number of '
                'generators should be larger than the threshold number of '
                'states for retraining.')

        model.deploy()
        states = []
        for i in range(self.niterations):
            for j, generator in enumerate(generators):
                if i > 0: # wait for previous to get latest model
                    index = (i - 1) * ngenerators + j
                    wait_for_it = [states[index]]
                else:
                    wait_for_it = []
                states.append(generator(model, reference, checks, wait_for_it))
            retrain_threshold = i * self.retrain_threshold
            for k, _ in enumerate(as_completed(states)):
                if k > retrain_threshold:
                    break # wait until at least retrain_threshold have completed
            try:
                completed = list(as_completed(states), timeout=2)
            except TimeoutError:
                pass
            logger.info('building dataset with {} states'.format(len(completed)))
            data = Dataset(completed)
            data_success = data.get(indices=data.success) # should be all of them?
            train, valid = get_train_valid_indices(
                    data_success.length(), # can be less than nstates
                    self.train_valid_split,
                    )
            data_train = data_success.get(indices=train)
            data_valid = data_success.get(indices=valid)
            data_train.log('data_train')
            data_valid.log('data_valid')
            model.train(data_train, data_valid, keep_deployed=True)
            delayed_deploy(model, model.model_future) # only deploy after training

            # save model obtained from this iteration
            model_ = model.copy()
            model_.deploy()
            save_state(
                    name='model_{}'.format(counter),
                    model=model_, # save model and its training data
                    data_train=data_train,
                    data_valid=data_valid,
                    require_done=False, # only completes when model is trained
                    )


@typeguard.typechecked
def load_learning(path_output: Union[Path, str]):
    path_output = Path(path_output)
    assert path_output.is_dir()
    classes = [
            SequentialLearning,
            ConcurrentLearning,
            None,
            ]
    for learning_cls in classes:
        assert learning_cls is not None, 'cannot find learning .yaml!'
        path_learning  = path_output / (learning_cls.__name__ + '.yaml')
        if path_learning.is_file():
            break
    with open(path_learning, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if 'WandBLogger' in config.keys():
        wandb_logger = WandBLogger(**config.pop('WandBLogger'))
    else:
        wandb_logger = None
    learning = learning_cls(wandb_logger=wandb_logger, **config)
    return learning
