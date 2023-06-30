from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import yaml
import logging

from concurrent.futures._base import TimeoutError
from concurrent.futures import as_completed
import parsl
from parsl.app.app import join_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.utils import get_train_valid_indices, save_yaml, copy_app_future
from psiflow.data import Dataset
from psiflow.wandb_utils import WandBLogger
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.sampling import BaseWalker, RandomWalker, PlumedBias, \
        BiasedDynamicWalker
from psiflow.generate import generate, generate_all
from psiflow.checks import Check, DiscrepancyCheck
from psiflow.state import save_state, load_state
from psiflow.utils import resolve_and_check


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
    use_formation_energy: bool = True
    atomic_energies: Optional[dict[str, float]] = None
    atomic_energies_box_size: float = 10
    train_from_scratch: bool = True
    niterations: int = 10
    num_tries_sampling: int = 5
    num_tries_reference: int = 1

    def __post_init__(self) -> None: # save self in output folder
        self.path_output = resolve_and_check(Path(self.path_output))
        if not self.path_output.is_dir():
            self.path_output.mkdir()
        config = asdict(self)
        config['path_output'] = str(self.path_output) # yaml requires str
        config.pop('wandb_logger')
        if self.wandb_logger is not None:
            config['WandBLogger'] = asdict(self.wandb_logger)
        path_config = self.path_output / (self.__class__.__name__ + '.yaml')
        if path_config.is_file():
            logger.warning('overriding learning config file {}'.format(path_config))
        save_yaml(config, outputs=[File(str(path_config))]).result()
        self.checks = []

    def output_exists(self, name):
        return (self.path_output / name).is_dir()

    def finish_iteration(
            self,
            name,
            model,
            walkers,
            data_train,
            data_valid,
            data_failed,
            require_done=False,
            ):
        save_state(
                self.path_output,
                name=name,
                model=model,
                walkers=walkers,
                data_train=data_train,
                data_valid=data_valid,
                data_failed=data_failed,
                require_done=require_done,
                )
        for check in self.checks:
            if isinstance(check, DiscrepancyCheck):
                if len(model.deploy_future) == 0:
                    model.deploy()
                check.update_model(model)
        if self.wandb_logger is not None:
            log = self.wandb_logger( # log training
                    run_name=name,
                    model=model,
                    data_train=data_train,
                    data_valid=data_valid,
                    )
            if require_done:
                log.result() # force execution

    def compute_atomic_energies(self, reference, data):
        @join_app
        def log_and_save_energies(path_learning, elements, *energies):
            logger.info('atomic energies:')
            for element, energy in zip(elements, energies):
                logger.info('\t{}: {} eV'.format(element, energy))
                if energy > 0:
                    logger.critical('\tatomic energy for element {} is {} but '
                            'should be negative'.format(element, energy))
                if energy == 1e10: # magic number to indicate SCF failed
                    raise ValueError('atomic energy calculation for element {}'
                            ' failed.'.format(element))
            with open(path_learning, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            config['atomic_energies'] = {el: float(en) for el, en in zip(elements, energies)}
            return save_yaml(config, outputs=[File(str(path_learning))])

        if self.atomic_energies is None:
            logger.info('computing energies of isolated atoms:')
            self.atomic_energies = {}
            for element in data.elements().result():
                energy = reference.compute_atomic_energy(
                        element,
                        self.atomic_energies_box_size,
                        )
                self.atomic_energies[element] = energy
        log_and_save_energies(
                self.path_output / (self.__class__.__name__ + '.yaml'),
                list(self.atomic_energies.keys()),
                *list(self.atomic_energies.values()),
                )

    def split_successful(self, data):
        data_success = data.get(indices=data.success)
        if self.use_formation_energy: # compute formation energies and add them as label
            data_success = data_success.set_formation_energy(**self.atomic_energies)
        assert data_success.length().result() > 0
        train, valid = get_train_valid_indices(
                data_success.length(), # can be less than nstates
                self.train_valid_split,
                )
        data_train = data_success.get(indices=train)
        data_valid = data_success.get(indices=valid)
        return data_train, data_valid

    def run_pretraining(
            self,
            model: BaseModel,
            reference: BaseReference,
            walkers: list[BaseWalker],
            ):
        nstates = self.pretraining_nstates
        amplitude_pos = self.pretraining_amplitude_pos
        amplitude_box = self.pretraining_amplitude_box
        logger.info('performing random pretraining')
        walkers_ = RandomWalker.multiply(
                nstates,
                data_start=Dataset([w.state_future for w in walkers]),
                amplitude_pos=amplitude_pos,
                amplitude_box=amplitude_box,
                )
        data = generate_all(walkers_, None, reference, 1, 1)
        states = []
        for i in range(nstates):
            index = i % len(walkers)
            walker = walkers[index]
            if isinstance(walker, BiasedDynamicWalker): # evaluate CVs!
                with_cv_inserted = walker.bias.evaluate(
                        data.get(indices=[i]),
                        as_dataset=True,
                        )
                states.append(with_cv_inserted[0])
            else:
                states.append(data[i])
        data = Dataset(states)
        data_train, data_valid = self.split_successful(data)
        data_train.log('data_train')
        data_valid.log('data_valid')
        model.initialize(data_train)
        model.train(data_train, data_valid)
        self.finish_iteration(
                'random_pretraining',
                model,
                walkers,
                data_train,
                data_valid,
                data_failed=None,
                require_done=True,
                )
        return data_train, data_valid

    def initialize_run(
            self,
            model: BaseModel,
            reference: BaseReference,
            walkers: list[BaseWalker],
            initial_data: Optional[Dataset] = None,
            ) -> tuple[Dataset, Dataset]:
        if self.wandb_logger is not None:
            self.wandb_logger.insert_name(model)
        if self.use_formation_energy:
            if not model.use_formation_energy and (model.model_future is not None):
                raise ValueError('model was initialized to train on absolute energies'
                                 ' but learning config is set to train on formation '
                                 'energy. ')
            if model.model_future is None:
                model.use_formation_energy = True
            logger.warning('model is trained on *formation* energy!')
            if initial_data is not None:
                self.compute_atomic_energies(reference, initial_data)
            else:
                self.compute_atomic_energies(
                        reference,
                        Dataset([w.state_future for w in walkers]),
                        )
        else:
            logger.warning('model is trained on *total* energy!')
        if model.model_future is None:
            if initial_data is None: # pretrain on random perturbations
                data_train, data_valid = self.run_pretraining(
                        model,
                        reference,
                        walkers,
                        )
            else: # pretrain on initial data
                data_train, data_valid = self.split_successful(initial_data)
                model.initialize(data_train)
                model.train(data_train, data_valid)
        else:
            if initial_data is None:
                data_train = Dataset([])
                data_valid = Dataset([])
            else:
                data_train, data_valid = self.split_successful(initial_data)
        return data_train, data_valid


@typeguard.typechecked
@dataclass
class SequentialLearning(BaseLearning):

    def run(
            self,
            model: BaseModel,
            reference: BaseReference,
            walkers: list[BaseWalker],
            initial_data: Optional[Dataset] = None,
            ) -> tuple[Dataset, Dataset]:
        data_train, data_valid = self.initialize_run(
                model,
                reference,
                walkers,
                initial_data,
                )
        model.deploy()
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            data = generate_all(
                    walkers,
                    model,
                    reference,
                    self.num_tries_sampling,
                    self.num_tries_reference,
                    checks=self.checks,
                    )
            _data_train, _data_valid = self.split_successful(data)
            data_train.append(_data_train)
            data_valid.append(_data_valid)
            data_train.log('data_train')
            data_valid.log('data_valid')
            if self.train_from_scratch:
                logger.info('reinitializing scale/shift/avg_num_neighbors on data_train')
                model.reset()
                model.initialize(data_train)
            model.train(data_train, data_valid)
            model.deploy()
            self.finish_iteration(
                    name=str(i),
                    model=model,
                    walkers=walkers,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    require_done=True,
                    )
        return data_train, data_valid


@typeguard.typechecked
@dataclass
class CommitteeLearning(BaseLearning):

    def run(
            self,
            model: BaseModel,
            reference: BaseReference,
            walkers: list[BaseWalker],
            initial_data: Optional[Dataset] = None,
            ) -> tuple[Dataset, Dataset]:
        data_train, data_valid = self.initialize_run(
                model,
                reference,
                walkers,
                initial_data,
                )
        model.deploy()
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            data = generate_all(
                    walkers,
                    model,
                    reference,
                    self.num_tries_sampling,
                    self.num_tries_reference,
                    checks=self.checks,
                    )
            _data_train, _data_valid = self.split_successful(data)
            data_train.append(_data_train)
            data_valid.append(_data_valid)
            data_train.log('data_train')
            data_valid.log('data_valid')
            if self.train_from_scratch:
                logger.info('reinitializing scale/shift/avg_num_neighbors on data_train')
                model.reset()
                model.initialize(data_train)
            model.train(data_train, data_valid)
            model.deploy()
            self.finish_iteration(
                    name=str(i),
                    model=model,
                    walkers=walkers,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    require_done=True,
                    )
        return data_train, data_valid


@join_app
def delayed_deploy(model, wait_for_it):
    model.deploy()


@typeguard.typechecked
@dataclass
class ConcurrentLearning(BaseLearning):
    min_states_per_iteration: int = 20
    max_states_per_iteration: int = 100

    def run(
            self,
            model: BaseModel,
            reference: BaseReference,
            walkers: list[BaseWalker],
            initial_data: Optional[Dataset] = None,
            ) -> tuple[Dataset, Dataset]:
        data_train, data_valid = self.initialize_run(
                model,
                reference,
                walkers,
                initial_data,
                )
        nwalkers = len(walkers)
        assert self.min_states_per_iteration <= nwalkers, ('the number of '
                'walkers should be larger than the threshold number of '
                'states for retraining.')
        states = []
        queue  = [None for i in range(nwalkers)]
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            model.deploy()
            j = 0
            assert self.max_states_per_iteration - len(states) >= self.min_states_per_iteration
            while len(states) < self.max_states_per_iteration:
                index = j % nwalkers
                state = generate(
                        str(index),
                        walkers[index],
                        model,
                        reference,
                        self.num_tries_sampling,
                        self.num_tries_reference,
                        queue[index],
                        checks=self.checks,
                        )
                queue[index] = state
                states.append(state)
                j += 1

            # finish previous training before gathering data
            model.model_future.result()

            # gather finished data, with a minimum of retrain_threshold
            retrain_threshold = (i + 1) * self.min_states_per_iteration
            for k, _ in enumerate(as_completed(states)):
                if k > retrain_threshold:
                    break # wait until at least retrain_threshold have completed
            completed = []
            try:
                for k, future in enumerate(as_completed(states, timeout=5)):
                    completed.append(future)
                    states.remove(future)
            except TimeoutError:
                pass
            logger.info('building dataset with {} states'.format(len(completed)))
            data = Dataset(completed)
            _data_train, _data_valid = self.split_successful(data)
            data_train.append(_data_train)
            data_valid.append(_data_valid)
            data_train.log('data_train')
            data_valid.log('data_valid')
            model.train(data_train, data_valid, keep_deployed=True)
            delayed_deploy(model, model.model_future) # only deploy after training

            # save model obtained from this iteration
            model_ = model.copy()
            model_.deploy()
            self.finish_iteration(
                    name=str(i),
                    model=model,
                    walkers=walkers,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    require_done=False,
                    )
        parsl.dfk().wait_for_current_tasks() # force execution of finish_iter
        return data_train, data_valid


@typeguard.typechecked
def load_learning(path_output: Union[Path, str]):
    path_output = resolve_and_check(Path(path_output))
    assert path_output.is_dir()
    classes = [
            SequentialLearning,
            ConcurrentLearning,
            IncrementalLearning,
            None,
            ]
    for learning_cls in classes:
        assert learning_cls is not None, 'cannot find learning .yaml!'
        path_learning  = path_output / (learning_cls.__name__ + '.yaml')
        if path_learning.is_file():
            break
    with open(path_learning, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['path_output'] = str(path_output)
    if 'WandBLogger' in config.keys():
        wandb_logger = WandBLogger(**config.pop('WandBLogger'))
    else:
        wandb_logger = None
    learning = learning_cls(wandb_logger=wandb_logger, **config)
    return learning
