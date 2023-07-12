from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Callable, Dict, Tuple
import typeguard
import logging
import yaml
import tempfile
from copy import deepcopy
from math import ceil
from pathlib import Path

import parsl
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.execution import ModelTrainingExecution, ModelEvaluationExecution
from psiflow.data import Dataset, app_join_dataset
from psiflow.utils import copy_app_future, save_yaml, copy_data_future, \
        resolve_and_check


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


@typeguard.typechecked
def evaluate_dataset(
        device: str,
        dtype: str,
        ncores: int,
        use_formation_energy: bool,
        load_calculator: Callable,
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> None:
    import torch
    import numpy as np
    from psiflow.data import read_dataset, save_dataset
    if device == 'cpu':
        torch.set_num_threads(ncores)
    if dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    dataset = read_dataset(slice(None), inputs=[inputs[0]])
    if len(dataset) > 0:
        calculator = load_calculator(inputs[1].filepath, device, dtype)
        for atoms in dataset:
            calculator.reset()
            atoms.calc = calculator
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            try: # some models do not have stress support
                stress = atoms.get_stress(voigt=False)
            except Exception as e:
                print(e)
                stress = np.zeros((3, 3))
            if use_formation_energy:
                atoms.info['formation_energy'] = energy
                atoms.info.pop('energy', None)
            else:
                atoms.info['energy'] = energy
                atoms.info.pop('formation_energy', None)
            atoms.info['stress'] = stress
            atoms.arrays['forces'] = forces
            atoms.calc = None
        save_dataset(dataset, outputs=[outputs[0]])


@typeguard.typechecked
class BaseModel:
    """Base Container for a trainable interaction potential"""

    def __init__(self, config: Dict) -> None:
        self.config_raw    = deepcopy(config)
        self.config_future = None
        self.model_future  = None
        self.deploy_future = {} # deployed models in float32 and float64

        # double-check whether required definitions are present
        context = psiflow.context()
        assert len(context[self.__class__]) == 2, ('Models require '
                'definition of both training and evaluation execution. '
                '{} only has the following definitions: {}'.format(
                    container,
                    self.definitions[container],
                    ))
        try: # initialize apps in context
            self.__class__.create_apps()
        except AssertionError: # apps already initialized; do nothing
            pass

    def train(
            self,
            training: Dataset,
            validation: Dataset,
            keep_deployed: bool = False,
            ) -> None:
        logger.info('training {} using {} states for training and {} for validation'.format(
            self.__class__.__name__,
            training.length().result(),
            validation.length().result(),
            ))
        if not keep_deployed:
            self.deploy_future = {} # no longer valid
        context = psiflow.context()
        future  = context.apps(self.__class__, 'train')( # new DataFuture instance
                self.config_future,
                inputs=[self.model_future, training.data_future, validation.data_future],
                outputs=[context.new_file('model_', '.pth')],
                )
        self.model_future = future.outputs[0]

    def initialize(self, dataset: Dataset) -> None:
        """Initializes the model based on a dataset"""
        assert self.config_future is None
        assert self.model_future is None
        self.deploy_future = {}
        available_labels = dataset.energy_labels().result()
        if self.use_formation_energy:
            assert 'formation_energy' in available_labels, ('key "{}" is not available in '
                    'dataset with keys {}'.format('formation_energy', available_labels))
        logger.info('initializing {} using dataset of {} states'.format(
            self.__class__.__name__, dataset.length().result()))
        context = psiflow.context()
        self.config_future = context.apps(self.__class__, 'initialize')( # to initialized config
                self.config_raw,
                inputs=[dataset.data_future],
                outputs=[context.new_file('model_', '.pth')],
                )
        self.model_future = self.config_future.outputs[0] # to undeployed model

    def evaluate(self, dataset: Dataset, batch_size: Optional[int] = 100) -> Dataset:
        """Evaluates a dataset using a model"""
        context = psiflow.context()
        length = dataset.length().result()
        if (batch_size is None) or (batch_size >= length):
            data_future = context.apps(self.__class__, 'evaluate')(
                    self.deploy_future,
                    self.use_formation_energy,
                    inputs=[dataset.data_future],
                    outputs=[context.new_file('data_', '.xyz')],
                    ).outputs[0]
            return Dataset(None, data_future=data_future)
        else:
            nbatches = ceil(length / batch_size)
            data_list = []
            for i in range(nbatches - 1):
                batch = dataset[i * batch_size : (i + 1) * batch_size]
                data_list.append(self.evaluate(batch))
            last = dataset[(nbatches - 1) * batch_size:]
            data_list.append(self.evaluate(last))
            data_future = app_join_dataset(
                    inputs=[d.data_future for d in data_list],
                    outputs=[context.new_file('data_', '.xyz')],
                    ).outputs[0]
            return Dataset(None, data_future=data_future)

    def reset(self) -> None:
        self.config_future = None
        self.model_future = None
        self.deploy_future = {}

    def save_deployed(
            self,
            path_deployed: Union[Path, str],
            dtype: str = 'float32',
            ) -> DataFuture:
        path_deployed = resolve_and_check(Path(path_deployed))
        return copy_data_future(
                inputs=[self.deploy_future[dtype]],
                outputs=[File(str(path_deployed))],
                ).outputs[0] # return data future

    def save(
            self,
            path: Union[Path, str],
            require_done: bool = True,
            ) -> Tuple[DataFuture, Optional[DataFuture], Optional[DataFuture]]:
        path = resolve_and_check(Path(path))
        path.mkdir(exist_ok=True)
        path_config_raw = path / (self.__class__.__name__ + '.yaml')
        future_raw = save_yaml(
                self.config_raw,
                outputs=[File(str(path_config_raw))],
                ).outputs[0]
        if self.config_future is not None:
            path_config = path / 'config_after_init.yaml'
            path_model  = path / 'model_undeployed.pth'
            future_config = save_yaml(
                    self.config_future,
                    outputs=[File(str(path_config))],
                    ).outputs[0]
            future_model = copy_data_future(
                    inputs=[self.model_future],
                    outputs=[File(str(path_model))],
                    ).outputs[0]
        else:
            future_config = None
            future_model  = None
        if require_done:
            future_raw.result()
            if self.config_future is not None:
                future_config.result()
                future_model.result()
        return future_raw, future_config, future_model

    def copy(self) -> BaseModel:
        context = psiflow.context()
        model = self.__class__(self.config_raw)
        if self.config_future is not None:
            model.config_future = copy_app_future(self.config_future)
            model.model_future = copy_data_future(
                    inputs=[self.model_future],
                    outputs=[context.new_file('model_', '.pth')],
                    ).outputs[0]
        if len(self.deploy_future) > 0:
            for key, future in self.deploy_future.items():
                model.deploy_future[key] = copy_data_future(
                        inputs=[self.deploy_future[key]],
                        outputs=[context.new_file('model_', '.pth')],
                        ).outputs[0]
        return model

    @property
    def use_formation_energy(self) -> bool:
        raise NotImplementedError

    @use_formation_energy.setter
    def use_formation_energy(self, arg) -> None:
        raise NotImplementedError

    @property
    def seed(self) -> int:
        raise NotImplementedError

    @seed.setter
    def seed(self, arg) -> None:
        raise NotImplementedError
