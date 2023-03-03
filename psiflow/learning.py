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
from parsl.app.app import join_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.utils import get_train_valid_indices, save_yaml
from psiflow.data import Dataset
from psiflow.wandb_utils import WandBLogger
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.sampling import RandomWalker, PlumedBias
from psiflow.generator import Generator
from psiflow.checks import Check, DiscrepancyCheck
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
    use_formation_energy: bool = True
    atomic_energies: Optional[dict[str, float]] = None
    atomic_energies_box_size: float = 10

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

    def finalize(
            self,
            name,
            model,
            generators,
            data_train,
            data_valid,
            data_failed,
            checks=None,
            require_done=False,
            ):
        save_state(
                self.path_output,
                name=name,
                model=model,
                generators=generators,
                data_train=data_train,
                data_valid=data_valid,
                data_failed=data_failed,
                require_done=False,
                )
        if checks is not None:
            for check in checks:
                if isinstance(check, DiscrepancyCheck):
                    if len(model.deploy_future) == 0:
                        model.deploy()
                    check.update_model(model)
        if self.wandb_logger is not None:
            if generators is not None:
                bias = generators[0].bias
            log = self.wandb_logger( # log training
                    run_name=name,
                    model=model,
                    generators=generators,
                    data_train=data_train,
                    data_valid=data_valid,
                    bias=bias,
                    )
            if require_done:
                log.result() # force execution

    def compute_atomic_energies(self, reference, data):
        @join_app
        def log_and_save_energies(path_learning, elements, *energies):
            logger.info('atomic energies:')
            for element, energy in zip(elements, energies):
                logger.info('\t{}: {} eV'.format(element, energy))
            with open(path_learning, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            config['atomic_energies'] = {el: en for el, en in zip(elements, energies)}
            return save_yaml(config, outputs=[File(path_config)])

        if self.atomic_energies is None:
            logger.info('computing energies of isolated atoms:')
            self.atomic_energies = {}
            for element in data.elements().result():
                energy = reference.get_atomic_energy(
                        element,
                        self.atomic_energies_box_size,
                        )
                self.atomic_energies[element] = energy
            log_and_save_energies(
                    self.path_output / (self.__class__.__name__ + '.yaml'),
                    list(self.atomic_energies.keys()),
                    *list(self.atomic_energies.values()),
                    )
        else:
            logger.info('found the following atomic energies in learning config')
            for element, energy in self.atomic_energies.items():
                logger.info('\t{}: {} eV'.format(element, energy.result()))

    def run_pretraining(
            self,
            model: BaseModel,
            reference: BaseReference,
            initial_data: Dataset,
            ):
        nstates = self.pretraining_nstates
        amplitude_pos = self.pretraining_amplitude_pos
        amplitude_box = self.pretraining_amplitude_box
        assert initial_data.length().result() > 0
        logger.info('performing random pretraining')
        if self.use_formation_energy:
            self.compute_atomic_energies(reference, initial_data)
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
        if self.use_formation_energy: # replace absolute with relative energies
            data_success = data_success.compute_formation_energy(**self.atomic_energies)
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
        self.finalize(
                name='random_pretraining',
                model=model,
                generators=generators,
                data_train=data_train,
                data_valid=data_valid,
                data_failed=data.get(indices=data.failed),
                require_done=True,
                )
        return data_train, data_valid


@typeguard.typechecked
@dataclass
class SequentialLearning(BaseLearning):
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
        if self.use_formation_energy:
            data = Dataset([g.walker.state_future for g in generators])
            self.compute_atomic_energies(reference, data)
        assert model.model_future is not None, ('model has to be initialized '
                'before running batch learning')
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            model.deploy()
            states = [g(model, reference, checks) for g in generators]
            data = Dataset(states)
            data_success = data.get(indices=data.success)
            if self.use_formation_energy: # replace absolute with relative energies
                data_success = data_success.compute_formation_energy(**self.atomic_energies)
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
            self.finalize(
                    name=str(i),
                    model=model,
                    generators=generators,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    checks=checks,
                    require_done=True,
                    )
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
        if self.use_formation_energy:
            data = Dataset([g.walker.state_future for g in generators])
            self.compute_atomic_energies(reference, data)
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
        queue  = [None for i in range(ngenerators)]
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            for j, generator in enumerate(generators):
                queued_future = queue[j]
                if queued_future is not None:
                    wait_for_it = [queued_future]
                else:
                    wait_for_it = []
                state = generator(model, reference, checks, wait_for_it)
                queue[j] = state
                states.append(state)

            # finish previous training before gathering data
            model.model_future.result()

            # gather finished data, with a minimum of retrain_threshold
            retrain_threshold = (i + 1) * self.retrain_threshold
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
            data_success = data.get(indices=data.success) # should be all of them?
            if self.use_formation_energy: # replace absolute with relative energies
                data_success = data_success.compute_formation_energy(**self.atomic_energies)
            train, valid = get_train_valid_indices(
                    data_success.length(), # can be less than nstates
                    self.train_valid_split,
                    )
            data_train.append(data_success.get(indices=train))
            data_valid.append(data_success.get(indices=valid))
            data_train.log('data_train')
            data_valid.log('data_valid')
            model.train(data_train, data_valid, keep_deployed=True)
            delayed_deploy(model, model.model_future) # only deploy after training

            # save model obtained from this iteration
            model_ = model.copy()
            model_.deploy()
            self.finalize(
                    name=str(i),
                    model=model,
                    generators=generators,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    checks=checks,
                    require_done=False,
                    )
        return data_train, data_valid


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
