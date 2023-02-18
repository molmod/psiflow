from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Callable, Dict, Tuple
import typeguard
import logging
import yaml
import tempfile
from copy import deepcopy
from pathlib import Path

import parsl
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.execution import ModelTrainingExecution, ModelEvaluationExecution
from psiflow.data import Dataset
from psiflow.utils import copy_app_future, save_yaml, copy_data_future


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


@typeguard.typechecked
def evaluate_dataset(
        device: str,
        dtype: str,
        ncores: int,
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
        atoms = dataset[0].copy()
        atoms.calc = load_calculator(inputs[1].filepath, device, dtype)
        for _atoms in dataset:
            _atoms.calc = None
            atoms.set_positions(_atoms.get_positions())
            atoms.set_cell(_atoms.get_cell())
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            try: # some models do not have stress support
                stress = atoms.get_stress(voigt=False)
            except Exception as e:
                print(e)
                stress = np.zeros((3, 3))
            _atoms.info['energy'] = energy
            _atoms.info['stress'] = stress
            _atoms.arrays['forces'] = forces
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
        raise NotImplementedError

    def evaluate(self, dataset: Dataset) -> Dataset:
        """Evaluates a dataset using a model"""
        context = psiflow.context()
        data_future = context.apps(self.__class__, 'evaluate')(
                self.deploy_future,
                inputs=[dataset.data_future],
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
        return copy_data_future(
                inputs=[self.deploy_future[dtype]],
                outputs=[File(str(path_deployed))],
                ).outputs[0] # return data future

    def save(
            self,
            path: Union[Path, str],
            require_done: bool = True,
            ) -> Tuple[DataFuture, Optional[DataFuture], Optional[DataFuture]]:
        path = Path(path)
        assert path.is_dir()
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
