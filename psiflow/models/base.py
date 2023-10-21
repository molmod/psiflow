from __future__ import annotations  # necessary for type-guarding class methods

import logging
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import typeguard
from parsl.app.app import join_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, app_join_dataset
from psiflow.utils import (copy_app_future, copy_data_future,
                           resolve_and_check, save_yaml)

logger = logging.getLogger(__name__)  # logging per module
logger.setLevel(logging.INFO)


@typeguard.typechecked
def evaluate_dataset(
    device: str,
    ncores: int,
    load_calculator: Callable,
    inputs: List[File] = [],
    outputs: List[File] = [],
) -> None:
    import numpy as np
    import torch

    from psiflow.data import NullState, read_dataset, write_dataset

    if device == "cpu":
        torch.set_num_threads(ncores)
    torch.set_default_dtype(torch.float32)
    dataset = read_dataset(slice(None), inputs=[inputs[0]])
    if len(dataset) > 0:
        calculator = load_calculator(inputs[1].filepath, device)
        for atoms in dataset:
            if atoms == NullState:
                continue
            calculator.reset()
            atoms.calc = calculator
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            try:  # some models do not have stress support
                stress = atoms.get_stress(voigt=False)
            except Exception as e:
                print(e)
                stress = np.zeros((3, 3))
            atoms.info["energy"] = energy
            atoms.info["stress"] = stress
            atoms.arrays["forces"] = forces
            atoms.calc = None
        write_dataset(dataset, outputs=[outputs[0]])


@join_app
@typeguard.typechecked
def evaluate_batched(
    model: BaseModel,
    dataset: Dataset,
    length: int,
    batch_size: int,
    outputs: list[File],
) -> AppFuture:
    context = psiflow.context()
    if (batch_size is None) or (batch_size >= length):
        future = context.apps(model.__class__, "evaluate")(
            inputs=[dataset.data_future, model.deploy_future],
            outputs=[outputs[0]],
        )
    else:
        nbatches = ceil(length / batch_size)
        data_list = []
        for i in range(nbatches - 1):
            batch = dataset[i * batch_size : (i + 1) * batch_size]
            f = context.apps(model.__class__, "evaluate")(
                inputs=[batch.data_future, model.deploy_future],
                outputs=[context.new_file("data_", ".xyz")],
            )
            data_list.append(f)
        last = dataset[(nbatches - 1) * batch_size :]
        f = context.apps(model.__class__, "evaluate")(
            inputs=[last.data_future, model.deploy_future],
            outputs=[context.new_file("data_", ".xyz")],
        )
        data_list.append(f)
        future = app_join_dataset(
            inputs=[d.outputs[0] for d in data_list],
            outputs=[outputs[0]],
        )
    return future


@join_app
def log_train(train_length, valid_length):
    logger.info(
        "training model using {} states for training and {} for validation".format(
            train_length,
            valid_length,
        )
    )
    return copy_app_future(0)


@typeguard.typechecked
class BaseModel:
    """Base Container for a trainable interaction potential"""

    def __init__(self, config: Dict) -> None:
        self.config_raw = deepcopy(config)
        self.config_future = None
        self.model_future = None
        self.deploy_future = None

        self.atomic_energies = {}

        # double-check whether required definitions are present
        assert len(psiflow.context()[self.__class__]) == 2
        try:  # initialize apps in context
            self.__class__.create_apps()
        except AssertionError:  # apps already initialized; do nothing
            pass

    def add_atomic_energy(self, element: str, energy: Union[float, AppFuture]) -> None:
        assert self.model_future is None, (
            "cannot add atomic energies after model has "
            "been initialized; reset model, add energy, and reinitialize"
        )
        if element in self.atomic_energies:
            if isinstance(energy, AppFuture):
                energy = energy.result()
            if isinstance(self.atomic_energies[element], AppFuture):
                existing = self.atomic_energies[element].result()
            assert energy == existing, (
                "model already has atomic energy "
                "for element {} ({}), which is different from {}"
                "".format(element, existing, energy)
            )
        self.atomic_energies[element] = energy

    def train(self, training: Dataset, validation: Dataset) -> None:
        log_train(training.length(), validation.length())
        inputs = [self.model_future]
        if self.do_offset:
            inputs += [
                training.subtract_offset(**self.atomic_energies).data_future,
                validation.subtract_offset(**self.atomic_energies).data_future,
            ]
        else:
            inputs += [
                training.data_future,
                validation.data_future,
            ]
        self.model_future, self.deploy_future = psiflow.context().apps(
            self.__class__, "train"
        )(
            self.config_future,
            inputs=inputs,
        )

    def initialize(self, dataset: Dataset) -> None:
        """Initializes the model based on a dataset"""
        assert self.config_future is None
        assert self.model_future is None
        if self.do_offset:
            inputs = [dataset.subtract_offset(**self.atomic_energies).data_future]
        else:
            inputs = [dataset.data_future]
        f, model_f, deploy_f = psiflow.context().apps(
            self.__class__, "initialize"
        )(  # to initialized config
            self.config_raw,
            inputs=inputs,
        )
        self.config_future = f
        self.model_future = model_f  # DataFuture
        self.deploy_future = deploy_f  # DataFuture

    def evaluate(self, dataset: Dataset, batch_size: Optional[int] = 100) -> Dataset:
        """Evaluates a dataset using a model"""
        future = evaluate_batched(
            self.copy(),
            dataset,
            dataset.length(),  # use join_app because length is unknown
            batch_size,
            outputs=[psiflow.context().new_file("data_", ".xyz")],
        )
        dataset = Dataset(None, data_future=future.outputs[0])
        if self.do_offset:
            return dataset.add_offset(**self.atomic_energies)
        else:
            return dataset

    def reset(self) -> None:
        self.config_future = None
        self.model_future = None
        self.deploy_future = None

    def save(
        self,
        path: Union[Path, str],
        require_done: bool = False,
    ) -> Tuple[DataFuture, Optional[DataFuture], Optional[DataFuture]]:
        path = resolve_and_check(Path(path))
        path.mkdir(exist_ok=True)
        path_config_raw = path / (self.__class__.__name__ + ".yaml")
        atomic_energies = {
            "atomic_energies_" + key: value
            for key, value in self.atomic_energies.items()
        }
        future_raw = save_yaml(
            deepcopy(self.config_raw),
            outputs=[File(str(path_config_raw))],
            **atomic_energies,
        ).outputs[0]
        if self.config_future is not None:
            path_config = path / "config_after_init.yaml"
            path_model = path / "model_undeployed.pth"
            path_deployed = path / "model_deployed.pth"
            future_config = save_yaml(
                self.config_future,
                outputs=[File(str(path_config))],
            ).outputs[0]
            future_model = copy_data_future(
                inputs=[self.model_future],
                outputs=[File(str(path_model))],
            ).outputs[0]
            future_deployed = copy_data_future(
                inputs=[self.deploy_future],
                outputs=[File(str(path_deployed))],
            ).outputs[0]
        else:
            future_config = None
            future_model = None
        if require_done:
            future_raw.result()
            if self.config_future is not None:
                future_config.result()
                future_model.result()
                future_deployed.result()
        return future_raw, future_config, future_model

    def copy(self) -> BaseModel:
        context = psiflow.context()
        model = self.__class__(self.config_raw)
        for element, energy in self.atomic_energies.items():
            model.add_atomic_energy(element, energy)
        if self.config_future is not None:
            model.config_future = copy_app_future(self.config_future)
            model.model_future = copy_data_future(
                inputs=[self.model_future],
                outputs=[context.new_file("model_", ".pth")],
            ).outputs[0]
            model.deploy_future = copy_data_future(
                inputs=[self.deploy_future],
                outputs=[context.new_file("model_", ".pth")],
            ).outputs[0]
        return model

    @property
    def do_offset(self) -> bool:
        return len(self.atomic_energies) > 0

    @property
    def seed(self) -> int:
        raise NotImplementedError

    @seed.setter
    def seed(self, arg) -> None:
        raise NotImplementedError
