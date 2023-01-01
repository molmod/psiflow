import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import wandb

from parsl.app.app import python_app

from flower.models import BaseModel, load_model
from flower.reference.base import BaseReference
from flower.sampling import RandomWalker
from flower.ensemble import Ensemble
from flower.data import Dataset
from flower.checks import Check, load_checks


@python_app(executors=['default'])
def log_data(
        table_name,
        wandb_name,
        wandb_group,
        wandb_project,
        visualize_structures,
        errors=None,
        error_labels=None,
        bias_labels=None,
        inputs=[],
        ):
    import wandb
    import tempfile
    from ase.data import chemical_symbols
    from ase.io import write
    from flower.data import read_dataset
    data = read_dataset(slice(None), inputs=[inputs[0]])
    wandb.init(
            project=wandb_project,
            group=wandb_group,
            job_type='dataset',
            name=wandb_name,
            )
    columns = [
            'index',
            'elements',
            'natoms',
            ]
    if error_labels is not None:
        if errors.shape[0] != len(data):
            raise AssertionError('error evaluation was not performed on every state')
        assert len(error_labels) == errors.shape[1]
        columns += error_labels
    if bias_labels is not None:
        assert len(bias_labels) == 2 * (len(inputs) - 1)
        for values in inputs[1:]:
            assert values.shape[0] == len(data)
            assert values.shape[1] == 2
        columns += bias_labels
    table_data = []
    for i, atoms in enumerate(data):
        elements = list(set([chemical_symbols[n] for n in atoms.numbers]))
        row = [i, ', '.join(elements), len(atoms)]
        if error_labels is not None:
            row += [e for e in errors[i, :]]
        if bias_labels is not None:
            for values in inputs[1:]:
                row.append(values[i, 0])
                row.append(values[i, 1])
        table_data.append(row)
    assert len(columns) == len(table_data[0])
    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({table_name: table})
    if visualize_structures:
        columns = ['index', 'structure']
        table_data = []
        for i, atoms in enumerate(data):
            tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
            tmp.close()
            path_pdb = tmp.name + '.pdb' # dummy log file
            write(path_pdb, atoms)
            table_data.append([i, wandb.Molecule(data_or_path=path_pdb)])
        table = wandb.Table(columns=columns, data=table_data)
        wandb.log({
            'structures for {}'.format(table_name): table,
            })
    wandb.finish()


@python_app(executors=['default'])
def log_ensemble(
        table_name,
        wandb_name,
        wandb_group,
        wandb_project,
        visualize_structures,
        errors=None,
        error_labels=None,
        bias_labels=None,
        inputs=[],
        ):
    import wandb
    import tempfile
    import numpy as np
    from ase.data import chemical_symbols
    from ase.io import write
    from flower.data import read_dataset
    data = read_dataset(slice(None), inputs=[inputs[0]])
    wandb.init(
            project=wandb_project,
            group=wandb_group,
            job_type='ensemble',
            name=wandb_name,
            )
    columns = [
            'walker index',
            'elements',
            'natoms',
            'tag',
            ]
    if error_labels is not None:
        if errors.shape[0] != len(data):
            raise AssertionError('error evaluation was not performed on every state')
        assert len(error_labels) == errors.shape[1]
        columns += error_labels
    if bias_labels is not None:
        assert len(bias_labels) % 2 == 0
        nvariables = len(bias_labels) // 2
        assert len(inputs[1:]) == len(data) * (nvariables + 1) # +1 due to tags
        columns += bias_labels
    table_data = []
    for i, atoms in enumerate(data):
        elements = list(set([chemical_symbols[n] for n in atoms.numbers]))
        tag_index = 1 + i
        row = [i, ', '.join(elements), len(atoms), inputs[tag_index]]
        if error_labels is not None:
            row += [e for e in errors[i, :]]
        if bias_labels is not None:
            for j in range(nvariables):
                index = int(1 + len(data) + i * nvariables + j) # casting necessary?
                if isinstance(inputs[index], np.ndarray): # False if bias not present
                    assert inputs[index].shape == (1, 2)
                    row.append(inputs[index][0, 0])
                    row.append(inputs[index][0, 1])
                else:
                    row.append(None)
                    row.append(None)
        table_data.append(row)
    table = wandb.Table(columns=columns, data=table_data, allow_mixed_types=True)
    wandb.log({table_name: table})
    wandb.finish()


class Manager:

    def __init__(self, path_output, wandb_project):
        self.path_output = Path(path_output)
        self.path_output.mkdir(parents=True, exist_ok=True)
        self.wandb_project = wandb_project
        self.iteration = 0

    def dry_run(
            self,
            model: BaseModel,
            reference: BaseReference,
            ensemble: Ensemble = None,
            random_walker: RandomWalker = None,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list] = None,
            ):
        context = model.context
        if random_walker is None:
            assert ensemble is not None
            random_walker = RandomWalker( # with default parameters
                    context,
                    ensemble.walkers[0].start_future,
                    )

        # single point evaluation
        evaluated = reference.evaluate(random_walker.state_future)
        evaluated.result()

        # generation of small dataset
        _ensemble = Ensemble.from_walker(random_walker, nwalkers=5)
        data = _ensemble.sample(7, model=None, checks=[])
        data = reference.evaluate(data)

        # short training and deploy
        model.reset()
        max_epochs = model.config_raw['max_epochs']
        model.config_raw['max_epochs'] = 2
        if (data_train is not None) and (data_valid is not None):
            assert data_train.length.result() >= 5 # only pass nonempty data
            assert data_valid.length.result() >= 2
            model.initialize(data_train)
            model.train(data_train, data_valid)
        new_train = data[:5]
        new_valid = data[5:]
        model.reset()
        model.initialize(new_train)
        model.train(new_train, new_valid)
        model.config_raw['max_epochs'] = max_epochs # revert to old max_epochs

        # deploy and propagate ensemble
        model.deploy()
        if ensemble is not None:
            data = ensemble.sample(
                    ensemble.nwalkers,
                    checks=checks,
                    model=model,
                    )
            data = reference.evaluate(data)
            assert data.length.result() == ensemble.nwalkers

        # save objects
        if ensemble is None:
            ensemble = _ensemble
        self.save( # test save
                prefix='dry_run',
                model=model,
                ensemble=ensemble,
                data_train=new_train,
                data_valid=new_valid,
                checks=checks,
                )

    def save(
            self,
            prefix,
            model: BaseModel,
            ensemble: Ensemble,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list] = None,
            ):
        path = self.path_output / prefix
        path.mkdir(parents=False, exist_ok=False) # parent should exist

        # model
        model.save(path)

        # ensemble
        path_ensemble = path / 'ensemble'
        path_ensemble.mkdir(parents=False)
        ensemble.save(path_ensemble)

        # data
        if data_train is not None:
            data_train.save(path / 'train.xyz')
        if data_valid is not None:
            data_valid.save(path / 'validate.xyz')

        # save checks if necessary
        if checks is not None:
            path_checks = path / 'checks'
            path_checks.mkdir(parents=False)
            for check in checks:
                check.save(path_checks) # all checks may be stored in same dir

    def load(self, prefix, context):
        path = self.path_output / prefix
        assert path.is_dir() # needs to exist

        # model
        model = load_model(context, path)

        # ensemble
        path_ensemble = path / 'ensemble'
        ensemble = Ensemble.load(context, path_ensemble)

        # data; optional
        path_train = path / 'train.xyz'
        if path_train.is_file():
            data_train = Dataset.load(context, path_train)
        else:
            data_train = Dataset(context)
        path_valid = path / 'validate.xyz'
        if path_valid.is_file():
            data_valid = Dataset.load(context, path_valid)
        else:
            data_valid = Dataset(context)

        # checks; optional
        path_checks = path / 'checks'
        if path_checks.is_dir():
            checks = load_checks(path_checks, context)
        return model, ensemble, data_train, data_valid, checks

    def log_dataset(
            self,
            wandb_name,
            wandb_group,
            dataset,
            visualize_structures=False,
            bias=None,
            model=None,
            error_kwargs=None,
            ):
        table_name = Path(dataset.data_future.filepath).name
        inputs = []
        if bias is not None:
            bias_labels = []
            for variable in bias.variables:
                inputs.append(bias.evaluate(dataset, variable=variable))
                bias_labels.append(variable)
                bias_labels.append('bias({})'.format(variable))
        else:
            bias_labels = None
        if model is not None:
            assert error_kwargs is not None
            _dataset = model.evaluate(dataset)
            errors = _dataset.get_errors(**error_kwargs)
            error_labels = [error_kwargs['metric'] + '_' + p for p in error_kwargs['properties']]
        else:
            errors = None
            error_labels = None
        return log_data(
                table_name,
                wandb_name,
                wandb_group,
                wandb_project=self.wandb_project,
                visualize_structures=visualize_structures,
                errors=errors,
                error_labels=error_labels,
                bias_labels=bias_labels,
                inputs=[dataset.data_future] + inputs,
                )

    def log_ensemble(
            self,
            wandb_name,
            wandb_group,
            ensemble,
            visualize_structures=False,
            model=None,
            error_kwargs=None,
            checks=None,
            ):
        table_name = 'ensemble'
        assert len(ensemble.walkers) > 0
        dataset = ensemble.as_dataset()
        if model is not None:
            assert error_kwargs is not None
            _dataset = model.evaluate(dataset)
            errors = _dataset.get_errors(**error_kwargs)
            error_labels = [error_kwargs['metric'] + '_' + p for p in error_kwargs['properties']]
        else:
            errors = None
            error_labels = None

        inputs = []

        # add walker tags to inputs
        for walker in ensemble.walkers:
            inputs.append(walker.tag_future)

        # add bias to inputs
        variables = []
        for bias in ensemble.biases:
            if bias is not None:
                variables += bias.variables
        variables = list(set(variables))
        if len(variables) > 0:
            bias_labels = []
            for variable in variables:
                bias_labels.append(variable)
                bias_labels.append('bias({})'.format(variable))
            for walker, bias in zip(ensemble.walkers, ensemble.biases): # evaluate bias per walker
                for i, variable in enumerate(variables):
                    if (bias is not None) and (variable in bias.variables):
                        inputs.append(bias.evaluate(
                            Dataset(bias.context, atoms_list=[walker.state_future]),
                            variable=variable,
                            ))
                    else:
                        inputs.append(False) # cannot pass None as input
        else:
            bias_labels = None

        # double check inputs contains tag info + bias info
        assert len(inputs) == len(ensemble.walkers) * (len(variables) + 1)
        return log_ensemble(
                table_name,
                wandb_name,
                wandb_group,
                wandb_project=self.wandb_project,
                visualize_structures=visualize_structures,
                errors=errors,
                error_labels=error_labels,
                bias_labels=bias_labels,
                inputs=[dataset.data_future] + inputs,
                )

    def increment(self):
        self.iteration += 1
