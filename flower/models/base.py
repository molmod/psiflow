import tempfile

from parsl.app.app import python_app
from parsl.data_provider.files import File

from flower.execution import ModelExecutionDefinition, Container
from flower.dataset import Dataset, _new_xyz


def evaluate_dataset(device, dtype, ncores, load_calculator, inputs=[], outputs=[]):
    import torch
    import numpy as np
    from flower.dataset import read_dataset, save_dataset
    if device == 'cpu':
        torch.set_num_threads(ncores)
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
            _atoms.info['energy_model'] = energy
            _atoms.info['stress_model'] = stress
            _atoms.arrays['forces_model'] = forces
        save_dataset(dataset, outputs=[outputs[0]])


def _new_deploy(context):
    _, name = tempfile.mkstemp(
            suffix='.pth',
            prefix='model_deployed_',
            dir=context.path,
            )
    return name


def _new_model(context):
    _, name = tempfile.mkstemp(
            suffix='.pth',
            prefix='model_',
            dir=context.path,
            )
    return name


class BaseModel(Container):
    """Base Container for a trainable interaction potential"""

    def __init__(self, context):
        super().__init__(context)

    def train(self, dataset):
        """Trains a model and returns it as an AppFuture"""
        raise NotImplementedError

    def evaluate(self, dataset):
        """Evaluates a dataset using a model and returns it as a covalent electron"""
        data_future = self.context.apps(self.__class__, 'evaluate')(
                inputs=[dataset.data_future, self.deploy_future],
                outputs=[File(_new_xyz(self.context))],
                ).outputs[0]
        return Dataset(self.context, data_future=data_future)
