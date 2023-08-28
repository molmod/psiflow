from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path
import numpy as np
import yaml
import logging

from ase.data import chemical_symbols

import parsl
from parsl.app.app import join_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.utils import save_yaml, copy_app_future
from psiflow.data import Dataset, FlowAtoms
from psiflow.models import BaseModel
from psiflow.committee import Committee
from psiflow.reference import BaseReference
from psiflow.walkers import BaseWalker, RandomWalker, \
        BiasedDynamicWalker
from psiflow.state import save_state, load_state
from psiflow.metrics import Metrics
from psiflow.sampling import sample_with_model, \
        sample_with_committee
from psiflow.utils import resolve_and_check


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
@dataclass
class BaseLearning:
    path_output: Union[Path, str]
    train_valid_split: float = 0.9
    pretraining_nstates: int = 50
    pretraining_amplitude_pos: float = 0.05
    pretraining_amplitude_box: float = 0.0
    metrics: Optional[Metrics] = Metrics()
    atomic_energies: dict[str, Union[float, AppFuture]] = field(default_factory=lambda: {})
    train_from_scratch: bool = True
    mix_training_validation: bool = True
    identifier: int = 0

    def __post_init__(self) -> None: # save self in output folder
        self.path_output = resolve_and_check(Path(self.path_output))
        if not self.path_output.is_dir():
            self.path_output.mkdir()
        atomic_energies = self.atomic_energies
        self.atomic_energies = {} # avoid errors in asdict
        config = asdict(self)
        self.atomic_energies = atomic_energies
        config['path_output'] = str(self.path_output) # yaml requires str
        config.pop('metrics')
        if self.metrics is not None:
            config['Metrics'] = self.metrics.as_dict()
        path_config = self.path_output / (self.__class__.__name__ + '.yaml')
        if path_config.is_file():
            logger.warning('overriding learning config file {}'.format(path_config))
        save_yaml(config, **atomic_energies, outputs=[File(str(path_config))]).result()

    def output_exists(self, name):
        return (self.path_output / name).is_dir()

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

        # double check whether atomic energies are present before
        # doing reference evaluations
        if model.do_offset:
            for element in Dataset(states).elements().result():
                assert element in model.atomic_energies

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
        states = [reference.evaluate(s) for s in states]
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
                    dataset=data,
                    )
        psiflow.wait()
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
        if len(self.atomic_energies) > 0:
            for element, energy in self.atomic_energies.items():
                model.add_atomic_energy(element, energy)
            logger.info('model contains atomic energy offsets')
        else:
            if len(model.atomic_energies) > 0:
                logger.warning('adding atomic energies from model into {}'.format(
                    self.__class__.__name__))
                ae = {e.result() if isinstance(e, AppFuture) else e for e in model.atomic_energies.values()}
                self.atomic_energies = ae
        if initial_data is not None:
            initial_data = initial_data.labeled()
            self.identifier = initial_data.assign_identifiers()
            if model.do_offset:
                for element in initial_data.elements().result():
                    assert element in model.atomic_energies
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


@typeguard.typechecked
@dataclass
class SequentialLearning(BaseLearning):
    initial_temperature: float = 100
    final_temperature: float = 600
    niterations: int = 10
    error_thresholds_for_reset: tuple[float, float] = (10, 200)

    def update_walkers(self, walkers: list[BaseWalker]):
        delta_beta = (1 / self.initial_temperature - 1 / self.final_temperature)
        delta_beta /= self.niterations
        for i, walker in enumerate(walkers):
            if not walker.is_reset().result():
                if hasattr(walker, 'temperature'):
                    T = 1 / (1 / walker.temperature - delta_beta)
                    if (T > 0) and (T < self.final_temperature):
                        walker.temperature = T
                    else: # reached max temp
                        walker.temperature = self.final_temperature

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
        # override initial temperature in walkers
        for walker in walkers:
            if hasattr(walker, 'temperature'):
                walker.temperature = self.initial_temperature
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
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
            psiflow.wait()
            self.update_walkers(walkers)
        return data


@typeguard.typechecked
@dataclass
class CommitteeLearning(SequentialLearning):
    nstates_per_iteration: int = 0

    def run(
            self,
            committee: Committee,
            reference: BaseReference,
            walkers: list[BaseWalker],
            initial_data: Optional[Dataset] = None,
            ) -> Dataset:
        assert self.nstates_per_iteration <= len(walkers)
        assert self.nstates_per_iteration > 0
        data = self.initialize_run(
                committee.models[0],
                reference,
                walkers,
                initial_data,
                )
        # override initial temperature in walkers
        for walker in walkers:
            if hasattr(walker, 'temperature'):
                walker.temperature = self.initial_temperature
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            new_data, self.identifier = sample_with_committee(
                    committee,
                    reference,
                    walkers,
                    self.identifier,
                    self.nstates_per_iteration,
                    self.error_thresholds_for_reset,
                    self.metrics,
                    )
            data = data + new_data
            data_train, data_valid = data.split(self.train_valid_split)
            committee.train(data_train, data_valid)
            save_state(
                    self.path_output,
                    str(i),
                    model=committee.models[0],
                    walkers=walkers,
                    data_train=data_train,
                    data_valid=data_valid,
                    )
            if self.metrics is not None:
                self.metrics.save(self.path_output / str(i), committee.models[0], data)
            committee.save(self.path_output / str(i) / 'committee')
            psiflow.wait()
            self.update_walkers(walkers)
        return data



@typeguard.typechecked
@dataclass
class IncrementalLearning(BaseLearning):
    cv_name: Optional[str] = None # have to be kwargs
    cv_start: Optional[float] = None
    cv_stop: Optional[float] = None
    cv_delta: Optional[float] = None
    niterations: int = 10
    error_thresholds_for_reset: tuple[float, float] = (10, 200)

    def update_walkers(self, walkers: list[BaseWalker]):
        for walker in walkers: # may not all contain bias
            if not hasattr(walker, 'bias'):
                continue
            if walker.is_reset().result():
                continue
            if not self.cv_name in walker.bias.variables:
                continue
            assert 'MOVINGRESTRAINT' in walker.bias.keys
            _, kappas, centers = walker.bias.get_moving_restraint(self.cv_name)
            steps = walker.steps
            new_centers = (
                    centers[1],
                    centers[1] + self.cv_delta,
                    )
            if np.sign(self.cv_stop - new_centers[1]) == np.sign(self.cv_delta):
                pass
            else: # start over
                new_centers = (self.cv_start, self.cv_start + self.cv_delta)
                walker.reset()
            walker.bias.adjust_moving_restraint(
                    self.cv_name,
                    steps=steps,
                    kappas=None, # don't change this
                    centers=new_centers,
                    )

    def run(
            self,
            model: BaseModel,
            reference: BaseReference,
            walkers: list[BaseWalker],
            initial_data: Optional[Dataset] = None,
            ) -> Dataset:
        assert self.cv_name is not None
        assert self.cv_start is not None
        assert self.cv_stop is not None
        assert self.cv_delta is not None
        data = self.initialize_run(
                model,
                reference,
                walkers,
                initial_data,
                )
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue # skip iterations in case of restarted run
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
            psiflow.wait()
            self.update_walkers(walkers)
        return data


@typeguard.typechecked
def load_learning(path_output: Union[Path, str]):
    path_output = resolve_and_check(Path(path_output))
    assert path_output.is_dir()
    classes = [
            SequentialLearning,
            CommitteeLearning,
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
    atomic_energies = {}
    for element in chemical_symbols:
        energy = config.pop('atomic_energies_' + element, None)
        if energy is not None:
            atomic_energies[element] = energy
    config['atomic_energies'] = atomic_energies
    config['path_output'] = str(path_output)
    if 'Metrics' in config.keys():
        metrics = Metrics(**config.pop('Metrics'))
    else:
        metrics = None
    learning = learning_cls(metrics=metrics, **config)
    return learning
