import yaml
import tempfile
from copy import deepcopy
from pathlib import Path

from parsl.app.app import python_app
from parsl.data_provider.files import File

from flower.execution import Container, ModelExecutionDefinition
from flower.data import Dataset, _new_file
from flower.utils import copy_app_future, save_yaml, copy_data_future


def evaluate_dataset(
        device,
        dtype,
        ncores,
        load_calculator,
        suffix,
        inputs=[],
        outputs=[],
        ):
    import torch
    import numpy as np
    from flower.data import read_dataset, save_dataset
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
            #sample.label(energy, forces, stress, log=None)
            _atoms.info['energy' + suffix] = energy
            _atoms.info['stress' + suffix] = stress
            _atoms.arrays['forces' + suffix] = forces
        save_dataset(dataset, outputs=[outputs[0]])


class BaseModel(Container):
    """Base Container for a trainable interaction potential"""

    def __init__(self, context, config):
        super().__init__(context)
        self.config_raw    = deepcopy(config)
        self.config_future = None
        self.model_future  = None
        self.deploy_future = {} # deployed models in float32 and float64

    def train(self, training, validation):
        """Trains a model and returns it as an AppFuture"""
        raise NotImplementedError

    def initialize(self, dataset):
        """Initializes the model based on a dataset"""
        raise NotImplementedError

    def evaluate(self, dataset, suffix='_model'):
        """Evaluates a dataset using a model and returns it as a covalent electron"""
        path_xyz = _new_file(self.context.path, 'data_', '.xyz')
        dtype = self.context[ModelExecutionDefinition].dtype
        assert dtype in self.deploy_future.keys()
        data_future = self.context.apps(self.__class__, 'evaluate')(
                suffix=suffix,
                inputs=[dataset.data_future, self.deploy_future[dtype]],
                outputs=[File(path_xyz)],
                ).outputs[0]
        return Dataset(self.context, data_future=data_future)

    def save(self, path, require_done=True):
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

    def copy(self):
        model = self.__class__(self.context, self.config_raw)
        if self.config_future is not None:
            model.config_future = copy_app_future(self.config_future)
            model.model_future = copy_data_future(
                    inputs=[self.model_future],
                    outputs=[File(_new_file(self.context.path, 'model_', '.pth'))],
                    ).outputs[0]
        if len(self.deploy_future) > 0:
            for key, future in self.deploy_future.items():
                model.deploy_future[key] = copy_data_future(
                        inputs=[self.deploy_future[key]],
                        outputs=[File(_new_file(self.context.path, 'model_', '.pth'))],
                        ).outputs[0]
        return model
