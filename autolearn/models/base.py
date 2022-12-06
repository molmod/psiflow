import tempfile

from parsl.app.app import python_app
from parsl.data_provider.files import File

from autolearn.execution import ModelExecutionDefinition


def evaluate_dataset(device, dtype, load_calculator, inputs=[], outputs=[]):
    import torch
    import numpy as np
    from autolearn.dataset import read_dataset, save_dataset
    #if device == 'cpu':
        #torch.set_num_threads(ncores)
    dataset = read_dataset(inputs=[inputs[0]])
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


class BaseModel:
    """Base class for a trainable interaction potential"""

    @staticmethod
    def load_calculator(path_model, device, dtype):
        raise NotImplementedError

    def train(self, dataset):
        """Trains a model and returns it as an AppFuture"""
        raise NotImplementedError

    def evaluate(self, dataset):
        """Evaluates a dataset using a model and returns it as a covalent electron"""
        definition = self.context[ModelExecutionDefinition]
        executor_label = definition.executor_label
        device         = definition.device
        dtype          = definition.dtype
        assert self.future_deploy is not None # needs to be deployed first
        p_evaluate_dataset = python_app(
                evaluate_dataset,
                executors=[executor_label],
                )
        dataset.future = p_evaluate_dataset(
                device,
                dtype,
                self.load_calculator,
                inputs=[dataset.future, self.future_deploy],
                outputs=[File(dataset.new_xyz())],
                ).outputs[0]

    def new_model(self):
        _, name = tempfile.mkstemp(
                suffix='.pth',
                prefix='model_',
                dir=self.context.path,
                )
        return name

    def new_deploy(self):
        _, name = tempfile.mkstemp(
                suffix='.pth',
                prefix='model_deployed_',
                dir=self.context.path,
                )
        return name

    @property
    def executor_label(self):
        return self.context[ModelExecutionDefinition].executor_label
