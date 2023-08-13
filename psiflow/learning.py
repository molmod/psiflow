from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import yaml
import logging

import parsl
from parsl.app.app import join_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.utils import save_yaml, copy_app_future
from psiflow.data import Dataset, FlowAtoms
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.walkers import BaseWalker, RandomWalker, \
        BiasedDynamicWalker
from psiflow.state import save_state, load_state
from psiflow.metrics import Metrics
from psiflow.sampling import sample_with_model
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
    metrics: Optional[Metrics] = None
    use_formation_energy: bool = True
    atomic_energies: Optional[dict[str, float]] = None
    atomic_energies_box_size: Optional[float] = None
    train_from_scratch: bool = True
    mix_training_validation: bool = True
    temperature_ramp: Optional[tuple[float, float]] = None # (Tmin, Tmax)
    error_thresholds_for_reset: tuple[float, float] = (10, 200)
    niterations: int = 10
    identifier: int = 0

    def __post_init__(self) -> None: # save self in output folder
        self.path_output = resolve_and_check(Path(self.path_output))
        if not self.path_output.is_dir():
            self.path_output.mkdir()
        config = asdict(self)
        config['path_output'] = str(self.path_output) # yaml requires str
        config.pop('metrics')
        if self.metrics is not None:
            config['Metrics'] = self.metrics.as_dict()
        path_config = self.path_output / (self.__class__.__name__ + '.yaml')
        if path_config.is_file():
            logger.warning('overriding learning config file {}'.format(path_config))
        save_yaml(config, outputs=[File(str(path_config))]).result()

    def output_exists(self, name):
        return (self.path_output / name).is_dir()

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

    def run_pretraining(
            self,
            model: BaseModel,
            reference: BaseReference,
            walkers: list[BaseWalker],
            ) -> Dataset:
        nstates = self.pretraining_nstates
        amplitude_pos = self.pretraining_amplitude_pos
        amplitude_box = self.pretraining_amplitude_box
        logger.info('performing random pretraining')
        walkers_ = RandomWalker.multiply(
                nstates,
                data_start=Dataset([w.state for w in walkers]),
                amplitude_pos=amplitude_pos,
                amplitude_box=amplitude_box,
                )
        states = [w.propagate().state for w in walkers_]
        for i in range(nstates):
            index = i % len(walkers)
            walker = walkers[index]
            if isinstance(walker, BiasedDynamicWalker): # evaluate CVs!
                with_cv_inserted = walker.bias.evaluate(
                        Dataset([states[i]]),
                        as_dataset=True,
                        )
                states[i] = with_cv_inserted[0]
            else:
                pass
        data = Dataset(states).labeled()
        self.identifier = data.assign_identifiers(self.identifier)
        data_train, data_valid = data.split(self.train_valid_split)
        model.initialize(data_train)
        model.train(data_train, data_valid)
        save_state(
                self.path_output,
                name='pretraining',
                model=model,
                walkers=walkers,
                data_train=data_train,
                data_valid=data_valid,
                )
        if self.metrics is not None:
            self.metrics.save(
                    self.path_output / 'pretraining',
                    model=model,
                    data=data,
                    )
        return data

    def initialize_run(
            self,
            model: BaseModel,
            reference: BaseReference,
            walkers: list[BaseWalker],
            initial_data: Optional[Dataset] = None,
            ) -> Dataset:
        """
        no initial data, model is None:
            do pretraining based on dataset of random perturbations on walker states

        no initial data, model is not None:
            continue with online learning, start with empty dataset

        initial data, model is None:
             train on initial data

        initial data, model is not None:
            continue with online learning, build on top of initial data

        """
        if self.metrics is not None:
            self.metrics.insert_name(model)
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
                        Dataset([w.state for w in walkers]),
                        )
        else:
            logger.warning('model is trained on *total* energy!')
        if initial_data is not None:
            initial_data = initial_data.labeled()
            self.identifier = initial_data.assign_identifiers()
        if model.model_future is None:
            if initial_data is None: # pretrain on random perturbations
                data = self.run_pretraining(
                        model,
                        reference,
                        walkers,
                        )
            else: # pretrain on initial data
                data_train, data_valid = initial_data.split(self.train_valid_split)
                model.initialize(data_train)
                model.train(data_train, data_valid)
                data = initial_data
        else:
            if initial_data is None:
                data = Dataset([])
            else:
                data = initial_data
        return data

    def set_temperature(self, walkers: list[BaseWalker], iteration: int):
        if self.temperature_ramp is not None:
            temperatures = 1 / np.linspace(
                    1 / self.temperature_ramp[0],
                    1 / self.temperature_ramp[1],
                    self.niterations,
                    )
            T = temperatures[iteration]
            for walker in walkers:
                if hasattr(walker, 'temperature'):
                    walker.temperature = T


@typeguard.typechecked
@dataclass
class SequentialLearning(BaseLearning):

    def run(
            self,
            model: BaseModel,
            reference: BaseReference,
            walkers: list[BaseWalker],
            initial_data: Optional[Dataset] = None,
            ) -> Dataset:
        data = self.initialize_run(
                model,
                reference,
                walkers,
                initial_data,
                )
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            self.set_temperature(walkers, i)
            new_data, self.identifier = sample_with_model(
                    model,
                    reference,
                    walkers,
                    self.identifier,
                    self.error_thresholds_for_reset,
                    self.metrics,
                    )
            data = data + new_data
            data_train, data_valid = data.split(self.train_valid_split)
            if self.train_from_scratch:
                logger.info('reinitializing scale/shift/avg_num_neighbors on data_train')
                model.reset()
                model.initialize(data_train)
            model.train(data_train, data_valid)
            save_state(
                    self.path_output,
                    str(i),
                    model=model,
                    walkers=walkers,
                    data_train=data_train,
                    data_valid=data_valid,
                    )
            if self.metrics is not None:
                self.metrics.save(self.path_output / str(i), model, data)
        return data


class IncrementalLearning:
    pass


@typeguard.typechecked
def load_learning(path_output: Union[Path, str]):
    path_output = resolve_and_check(Path(path_output))
    assert path_output.is_dir()
    classes = [
            SequentialLearning,
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
    if 'Metrics' in config.keys():
        metrics = Metrics(**config.pop('Metrics'))
    else:
        metrics = None
    learning = learning_cls(metrics=metrics, **config)
    return learning
