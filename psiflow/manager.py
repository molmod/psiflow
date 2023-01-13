from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Callable, Dict, Tuple, Any
import typeguard
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import wandb
import numpy as np

from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.execution import ExecutionContext
from psiflow.models import BaseModel, load_model
from psiflow.reference.base import BaseReference
from psiflow.sampling import RandomWalker, PlumedBias
from psiflow.ensemble import Ensemble
from psiflow.data import Dataset
from psiflow.checks import Check, load_checks, SafetyCheck
from psiflow.utils import copy_app_future, log_data_to_wandb


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


@typeguard.typechecked
def _app_log_data(
        errors: Optional[np.ndarray] = None,
        error_labels: Optional[List[str]] = None,
        bias_labels: Optional[List[str]] = None,
        inputs: List[Union[File, np.ndarray]] = [],
        ) -> List[List]:
    from ase.data import chemical_symbols
    from ase.io import write
    from psiflow.data import read_dataset
    data = read_dataset(slice(None), inputs=[inputs[0]])
    columns = ['index', 'location', 'elements', 'natoms']
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
    location = Path(inputs[0].filepath).name
    for i, atoms in enumerate(data):
        elements = list(set([chemical_symbols[n] for n in atoms.numbers]))
        row = [i, location, ', '.join(elements), len(atoms)]
        if error_labels is not None:
            row += [e for e in errors[i, :]]
        if bias_labels is not None:
            for values in inputs[1:]:
                row.append(values[i, 0])
                row.append(values[i, 1])
        assert len(columns) == len(row)
        table_data.append(row)
    return [columns] + table_data
app_log_data = python_app(_app_log_data, executors=['default'])


@typeguard.typechecked
def _app_log_ensemble(
        errors: Optional[np.ndarray] = None,
        error_labels: Optional[List[str]] = None,
        bias_labels: Optional[List[str]] = None,
        inputs: List[Union[File, np.ndarray, str, bool]] = [],
        ) -> List[List]:
    import numpy as np
    from ase.data import chemical_symbols
    from ase.io import write
    from psiflow.data import read_dataset
    data = read_dataset(slice(None), inputs=[inputs[0]])
    columns = ['walker index', 'elements', 'natoms', 'tag']
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
        assert len(columns) == len(row)
        table_data.append(row)
    return [columns] + table_data
app_log_ensemble = python_app(_app_log_ensemble, executors=['default'])


@typeguard.typechecked
def log_data(
        dataset: Dataset,
        bias: Optional[PlumedBias],
        model: Optional[BaseModel],
        error_kwargs: Optional[dict[str, Any]],
        ) -> AppFuture:
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
        if len(model.deploy_future) == 0:
            model.deploy()
        #_dataset = model.evaluate(dataset)
        errors = Dataset.get_errors(
                dataset,
                model.evaluate(dataset),
                **error_kwargs,
                )
        error_labels = [error_kwargs['metric'] + '_' + p for p in error_kwargs['properties']]
    else:
        errors = None
        error_labels = None
    return app_log_data(
            errors=errors,
            error_labels=error_labels,
            bias_labels=bias_labels,
            inputs=[dataset.data_future] + inputs,
            )


@typeguard.typechecked
def log_ensemble(ensemble: Ensemble) -> AppFuture:
    assert len(ensemble.walkers) > 0
    dataset = ensemble.as_dataset()
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
                        Dataset(bias.context, [walker.state_future]),
                        variable=variable,
                        ))
                else:
                    inputs.append(False) # cannot pass None as input
    else:
        bias_labels = None

    # double check inputs contains tag info + bias info
    assert len(inputs) == len(ensemble.walkers) * (len(variables) + 1)
    return app_log_ensemble(
            bias_labels=bias_labels,
            inputs=[dataset.data_future] + inputs,
            )


@typeguard.typechecked
class Manager:

    def __init__(
            self,
            path_output: Union[Path, str],
            wandb_project: str,
            wandb_group: str,
            restart: bool = False,
            error_kwargs: Optional[dict[str, Any]] = None,
            error_x_axis: Optional[str] = 'index',
            ) -> None:
        self.path_output = Path(path_output)
        self.restart = restart
        self.wandb_project = wandb_project
        self.wandb_group   = wandb_group
        if error_kwargs is None:
            error_kwargs = {
                    'metric': 'mae',
                    'properties': ['energy', 'forces', 'stress'],
                    }
        self.error_kwargs = error_kwargs
        self.error_x_axis = error_x_axis
        # output directory can only exist if this run is a restart
        self.path_output.mkdir(parents=True, exist_ok=self.restart)

    def dry_run(
            self,
            model: BaseModel,
            reference: BaseReference,
            ensemble: Optional[Ensemble] = None,
            random_walker: Optional[RandomWalker] = None,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list[Check]] = None,
            ) -> None:
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
        if checks is None:
            checks = [SafetyCheck()]
        else:
            checks = checks + [SafetyCheck()]
        _ensemble.walkers[3].tag_future = copy_app_future('unsafe')
        data = _ensemble.sample(7, model=None, checks=checks)
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
            assert data.length().result() == ensemble.nwalkers

        # log and save objects
        if ensemble is None:
            ensemble = _ensemble
        log = self.log_wandb(
                'dry_run',
                model,
                ensemble,
                data_train=new_train,
                data_valid=new_valid,
                checks=checks,
                )
        log.result()
        self.save( # test save
                name='dry_run',
                model=model,
                ensemble=ensemble,
                data_train=new_train,
                data_valid=new_valid,
                checks=checks,
                )

    def save(
            self,
            name: str,
            model: BaseModel,
            ensemble: Ensemble,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            data_failed: Optional[Dataset] = None,
            checks: Optional[list[Check]] = None,
            ) -> None:
        path = self.path_output / name
        path.mkdir(parents=False, exist_ok=self.restart) # parent should exist

        # model
        model.save(path)

        # ensemble
        path_ensemble = path / 'ensemble'
        path_ensemble.mkdir(parents=False, exist_ok=self.restart)
        ensemble.save(path_ensemble, restart=self.restart)

        # data
        if data_train is not None:
            data_train.save(path / 'train.xyz')
        if data_valid is not None:
            data_valid.save(path / 'validate.xyz')
        if data_failed is not None:
            data_failed.save(path / 'failed.xyz')

        # save checks if necessary
        if checks is not None:
            path_checks = path / 'checks'
            path_checks.mkdir(parents=False, exist_ok=self.restart)
            for check in checks:
                check.save(path_checks) # all checks may be stored in same dir

    def load(
            self,
            name: str,
            context: ExecutionContext,
            ) -> Tuple[
                    BaseModel,
                    Ensemble,
                    Optional[Dataset],
                    Optional[Dataset],
                    Optional[list[Check]],
                    ]:
        path = self.path_output / name
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
            data_train = Dataset(context, []) # empty dataset
        path_valid = path / 'validate.xyz'
        if path_valid.is_file():
            data_valid = Dataset.load(context, path_valid)
        else:
            data_valid = Dataset(context, [])

        # checks; optional
        path_checks = path / 'checks'
        if path_checks.is_dir():
            checks = load_checks(path_checks, context)
        return model, ensemble, data_train, data_valid, checks

    def log_wandb(
            self,
            run_name: str,
            model: BaseModel,
            ensemble: Ensemble,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            data_failed: Optional[Dataset] = None,
            checks: Optional[list[Check]] = None,
            bias: Optional[PlumedBias] = None,
            ) -> AppFuture:
        log_futures = {}
        logger.info('logging data to wandb')
        if data_train is not None:
            log_futures['training'] = log_data( # log training and validation data as tables
                    dataset=data_train,
                    bias=bias,
                    model=model,
                    error_kwargs=self.error_kwargs,
                    )
        if data_valid is not None:
            log_futures['validation'] = log_data(
                    dataset=data_valid,
                    bias=bias,
                    model=model,
                    error_kwargs=self.error_kwargs,
                    )
        if data_failed is not None:
            log_futures['failed'] = log_data( # log states with failed reference calculation
                    dataset=data_failed,
                    bias=bias,
                    model=model,
                    error_kwargs=self.error_kwargs,
                    )
        if ensemble is not None:
            log_futures['ensemble'] = log_ensemble(ensemble=ensemble)
        if checks is not None:
            for check in checks:
                name = check.__class__.__name__
                dataset = Dataset(model.context, check.states)
                log_futures[name] = log_data(
                        dataset,
                        bias=bias,
                        model=None,
                        error_kwargs=None,
                        ) # no model or error kwargs
        return log_data_to_wandb(
                run_name,
                self.wandb_group,
                self.wandb_project,
                self.error_x_axis,
                list(log_futures.keys()),
                inputs=list(log_futures.values()),
                )
