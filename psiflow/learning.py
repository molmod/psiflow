from __future__ import annotations  # necessary for type-guarding class methods

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
import yaml
from ase.data import chemical_symbols
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.committee import Committee
from psiflow.data import Dataset
from psiflow.metrics import Metrics
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.sampling import sample_with_committee, sample_with_model
from psiflow.state import save_state
from psiflow.utils import apply_temperature_ramp, resolve_and_check, save_yaml
from psiflow.walkers import BaseWalker, BiasedDynamicWalker, RandomWalker

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
@dataclass
class BaseLearning:
    path_output: Union[Path, str]
    train_valid_split: float = 0.9
    pretraining_nstates: int = 50
    pretraining_amplitude_pos: float = 0.05
    pretraining_amplitude_box: float = 0.0
    metrics: Metrics = field(default_factory=lambda: Metrics())  # mutable default
    atomic_energies: dict[str, Union[float, AppFuture]] = field(
        default_factory=lambda: {}
    )
    train_from_scratch: bool = True
    mix_training_validation: bool = True
    identifier: int = 0

    def __post_init__(self) -> None:  # save self in output folder
        self.path_output = resolve_and_check(Path(self.path_output))
        self.path_output.mkdir(parents=True, exist_ok=True)
        atomic_energies = self.atomic_energies
        self.atomic_energies = {}  # avoid errors in asdict
        config = asdict(self)
        self.atomic_energies = atomic_energies
        config["path_output"] = str(self.path_output)  # yaml requires str
        config.pop("metrics")
        config["Metrics"] = self.metrics.as_dict()
        path_config = self.path_output / (self.__class__.__name__ + ".yaml")
        if path_config.is_file():
            logger.warning("overriding learning config file {}".format(path_config))
        save_yaml(config, **atomic_energies, outputs=[File(str(path_config))]).result()

    def output_exists(self, name):
        return (self.path_output / name).is_dir()

    def run_pretraining(
        self,
        model: BaseModel,
        reference: BaseReference,
        walkers: list[BaseWalker],
    ) -> Dataset:
        self.metrics.iteration = "pretraining"
        nstates = self.pretraining_nstates
        amplitude_pos = self.pretraining_amplitude_pos
        amplitude_box = self.pretraining_amplitude_box
        logger.info("performing random pretraining")
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
            if isinstance(walker, BiasedDynamicWalker):  # evaluate CVs!
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
            name="pretraining",
            model=model,
            walkers=walkers,
            data_train=data_train,
            data_valid=data_valid,
        )
        self.metrics.save(
            self.path_output / "pretraining",
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
        self.metrics.insert_name(model)
        if len(self.atomic_energies) > 0:
            for element, energy in self.atomic_energies.items():
                model.add_atomic_energy(element, energy)
            logger.info("model contains atomic energy offsets")
        else:
            if len(model.atomic_energies) > 0:
                logger.warning(
                    "adding atomic energies from model into {}".format(
                        self.__class__.__name__
                    )
                )
                ae = {
                    s: e.result() if isinstance(e, AppFuture) else e
                    for s, e in model.atomic_energies.items()
                }
                for element, energy in ae.items():
                    assert (
                        energy < 1e10
                    ), "atomic energy calculation for element {} failed!".format(
                        element
                    )
                self.atomic_energies = ae
        if initial_data is not None:
            initial_data = initial_data.labeled()
            self.identifier = initial_data.assign_identifiers()
            if model.do_offset:
                for element in initial_data.elements().result():
                    assert element in model.atomic_energies
        if model.model_future is None:
            if initial_data is None:  # pretrain on random perturbations
                data = self.run_pretraining(
                    model,
                    reference,
                    walkers,
                )
            else:  # pretrain on initial data
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
    temperature_ramp: Optional[tuple[float]] = None
    niterations: int = 10
    error_thresholds_for_reset: tuple[float, float] = (10, 200)

    def update_walkers(self, walkers: list[BaseWalker], initialize=False):
        if self.temperature_ramp is not None:
            for walker in walkers:
                if hasattr(walker, "temperature"):
                    if initialize:
                        walker.temperature = self.temperature_ramp[
                            0
                        ]  # initial temperature
                    else:
                        temperature = apply_temperature_ramp(
                            *self.temperature_ramp,
                            walker.temperature,
                        )
                        if not walker.is_reset().result():
                            walker.temperature = temperature

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
                continue  # skip iterations in case of restarted run
            self.metrics.iteration = i
            self.update_walkers(walkers, initialize=(i == 0))
            new_data, self.identifier = sample_with_model(
                model,
                reference,
                walkers,
                self.identifier,
                self.error_thresholds_for_reset,
                self.metrics,
            )
            assert new_data.length().result() > 0, "no new states were generated!"
            data = data + new_data
            data_train, data_valid = data.split(self.train_valid_split)
            if self.train_from_scratch:
                logger.info(
                    "reinitializing scale/shift/avg_num_neighbors on data_train"
                )
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
            self.metrics.save(self.path_output / str(i), model, data)
            psiflow.wait()
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
        assert initial_data is not None
        data = initial_data
        committee.train(*data.shuffle().split(0.9))  # initial training
        for i in range(self.niterations):
            if self.output_exists(str(i)):
                continue  # skip iterations in case of restarted run
            self.metrics.iteration = i
            self.update_walkers(walkers, initialize=(i == 0))
            new_data, self.identifier = sample_with_committee(
                committee,
                reference,
                walkers,
                self.identifier,
                self.nstates_per_iteration,
                self.error_thresholds_for_reset,
                self.metrics,
            )
            assert new_data.length().result() > 0, "no new states were generated!"
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
            self.metrics.save(self.path_output / str(i), committee.models[0], data)
            committee.save(self.path_output / str(i) / "committee")
            psiflow.wait()
        return data


@typeguard.typechecked
@dataclass
class IncrementalLearning(BaseLearning):
    cv_name: Optional[str] = None  # have to be kwargs
    cv_start: Optional[float] = None
    cv_stop: Optional[float] = None
    cv_delta: Optional[float] = None
    niterations: int = 10
    error_thresholds_for_reset: tuple[float, float] = (10, 200)

    def update_walkers(self, walkers: list[BaseWalker], initialize=False):
        for walker in walkers:  # may not all contain bias
            if not hasattr(walker, "bias"):
                continue
            if self.cv_name not in walker.bias.variables:
                continue
            assert "MOVINGRESTRAINT" in walker.bias.keys
            _, kappas, centers = walker.bias.get_moving_restraint(self.cv_name)
            steps = walker.steps
            if initialize or walker.is_reset().result():
                new_centers = (self.cv_start, self.cv_start + self.cv_delta)
            else:
                new_centers = (
                    centers[1],
                    centers[1] + self.cv_delta,
                )
                check_interval = (self.cv_stop - new_centers[1]) * np.sign(
                    self.cv_delta
                ) >= 0
                if check_interval:
                    pass
                else:  # start over
                    new_centers = (self.cv_start, self.cv_start + self.cv_delta)
                    walker.reset()
            walker.bias.adjust_moving_restraint(
                self.cv_name,
                steps=steps,
                kappas=None,  # don't change this
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
                continue  # skip iterations in case of restarted run
            self.metrics.iteration = i
            self.update_walkers(walkers, initialize=(i == 0))
            new_data, self.identifier = sample_with_model(
                model,
                reference,
                walkers,
                self.identifier,
                self.error_thresholds_for_reset,
                self.metrics,
            )
            assert new_data.length().result() > 0, "no new states were generated!"
            data = data + new_data
            data_train, data_valid = data.split(self.train_valid_split)
            if self.train_from_scratch:
                logger.info(
                    "reinitializing scale/shift/avg_num_neighbors on data_train"
                )
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
            self.metrics.save(self.path_output / str(i), model, data)
            psiflow.wait()
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
        assert learning_cls is not None, "cannot find learning .yaml!"
        path_learning = path_output / (learning_cls.__name__ + ".yaml")
        if path_learning.is_file():
            break
    with open(path_learning, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    atomic_energies = {}
    for element in chemical_symbols:
        energy = config.pop("atomic_energies_" + element, None)
        if energy is not None:
            atomic_energies[element] = energy
    config["atomic_energies"] = atomic_energies
    config["path_output"] = str(path_output)
    metrics = Metrics(**config.pop("Metrics", {}))
    learning = learning_cls(metrics=metrics, **config)
    return learning
