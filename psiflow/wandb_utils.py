from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, Callable, Dict, Tuple, Any
import typeguard
import os
import logging
from pathlib import Path
import numpy as np

import parsl
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.models import BaseModel, NequIPModel
from psiflow.reference.base import BaseReference
from psiflow.sampling import RandomWalker, PlumedBias
from psiflow.generator import Generator
from psiflow.data import Dataset
from psiflow.checks import Check
from psiflow.utils import copy_app_future


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
def _app_log_data(
        errors: Optional[np.ndarray] = None,
        error_labels: Optional[list[str]] = None,
        bias_labels: Optional[list[str]] = None,
        inputs: list[Union[File, np.ndarray]] = [],
        ) -> list[list]:
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
def _app_log_generators(
        generator_names: list[str],
        errors: Optional[np.ndarray] = None,
        error_labels: Optional[list[str]] = None,
        bias_labels: Optional[list[str]] = None,
        inputs: list[Union[File, np.ndarray, str, bool, int]] = [],
        ) -> list[list]:
    import numpy as np
    from ase.data import chemical_symbols
    from ase.io import write
    from psiflow.data import read_dataset
    data = read_dataset(slice(None), inputs=[inputs[0]])
    columns = ['generator name', 'elements', 'natoms', 'counter']
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
        name = generator_names[i]
        row = [name, ', '.join(elements), len(atoms), inputs[tag_index]]
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
app_log_generators = python_app(_app_log_generators, executors=['default'])


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
def log_generators(generators: list[Generator]) -> AppFuture:
    assert len(generators) > 0
    states = [generator.walker.state_future for generator in generators]
    dataset = Dataset(states)
    inputs = []

    # add walker counters to inputs
    generator_names = [g.name for g in generators]
    for g in generators:
        inputs.append(g.walker.counter_future)

    # add bias to inputs
    variables = []
    for g in generators:
        if g.bias is not None:
            variables += g.bias.variables
    variables = list(set(variables))
    if len(variables) > 0:
        bias_labels = []
        for variable in variables:
            bias_labels.append(variable)
            bias_labels.append('bias({})'.format(variable))
        for g in generators:
            for i, variable in enumerate(variables):
                if (g.bias is not None) and (variable in g.bias.variables):
                    inputs.append(g.bias.evaluate(
                        Dataset([g.walker.state_future]),
                        variable=variable,
                        ))
                else:
                    inputs.append(False) # cannot pass None as input
    else:
        bias_labels = None

    # double check inputs contains tag info + bias info
    assert len(inputs) == len(generators) * (len(variables) + 1)
    return app_log_generators(
            generator_names=generator_names,
            bias_labels=bias_labels,
            inputs=[dataset.data_future] + inputs,
            )


@typeguard.typechecked
class WandBLogger:

    def __init__(
            self,
            wandb_project: str,
            wandb_group: str,
            error_kwargs: Optional[dict[str, Any]] = None,
            error_x_axis: Optional[str] = 'index',
            ) -> None:
        self.wandb_project = wandb_project
        self.wandb_group   = wandb_group
        if error_kwargs is None:
            error_kwargs = {
                    'metric': 'mae',
                    'properties': ['energy', 'forces', 'stress'],
                    }
        self.error_kwargs = error_kwargs
        self.error_x_axis = error_x_axis

    def __call__(
            self,
            run_name: str,
            model: BaseModel,
            generators: list[Generator],
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
        if generators is not None:
            log_futures['generators'] = log_generators(generators=generators)
        if checks is not None:
            for check in checks:
                name = check.__class__.__name__
                dataset = Dataset(check.states)
                log_futures[name] = log_data(
                        dataset,
                        bias=bias,
                        model=None,
                        error_kwargs=None,
                        ) # no model or error kwargs
        if bias is None: # if bias is not available, u
            error_x_axis = 'index'
        else:
            assert self.error_x_axis in bias.variables, ('wandb logging is '
                    'supposed to use {} as x axis for plotting errors, but '
                    'supplied bias only has the following variables: {}'.format(
                        self.error_x_axis,
                        bias.variables,
                        ))
            error_x_axis = self.error_x_axis
        logger.info('\twandb project: {}'.format(self.wandb_project))
        logger.info('\twandb group  : {}'.format(self.wandb_group))
        logger.info('\twandb name   : {}'.format(run_name))
        logger.info('\tx axis       : {}'.format(error_x_axis))
        for key in log_futures.keys():
            logger.info('\t\t{}'.format(key))
        return to_wandb(
                run_name,
                self.wandb_group,
                self.wandb_project,
                error_x_axis,
                list(log_futures.keys()),
                inputs=list(log_futures.values()),
                )

    def insert_name(self, model: BaseModel) -> None:
        if isinstance(model, NequIPModel):
            model.config_raw['wandb_group'] = self.wandb_group
        else:
            logger.warning('cannot set wandb name for model {}'.format(model.__class__))

    def parameters(self):
        return {
                'wandb_project': self.wandb_project,
                'wandb_group': self.wandb_group,
                'error_kwargs': self.error_kwargs,
                'error_x_axis': self.error_x_axis,
                }


@typeguard.typechecked
def _to_wandb(
        run_name: str,
        group: str,
        project: str,
        error_x_axis: str,
        names: list[str],
        inputs: list[list[list]] = [], # list of 2D tables
        ) -> None:
    from pathlib import Path
    import tempfile
    import wandb
    path_wandb = Path(tempfile.mkdtemp())
    wandb.init(
            name=run_name,
            group=group,
            project=project,
            resume='allow',
            dir=path_wandb,
            )
    wandb_log = {}
    assert len(names) == len(inputs)
    for name, data in zip(names, inputs):
        table = wandb.Table(columns=data[0], data=data[1:])
        if name in ['training', 'validation', 'failed']:
            errors_to_plot = [] # check which error labels are present
            for l in data[0]:
                if l.endswith('energy') or l.endswith('forces') or l.endswith('stress'):
                    errors_to_plot.append(l)
            assert error_x_axis in data[0]
            for error in errors_to_plot:
                title = name + '_' + error
                wandb_log[title] = wandb.plot.scatter(
                        table,
                        error_x_axis,
                        error,
                        title=title,
                        )
        else:
            wandb_log[name + '_table'] = table
    assert path_wandb.is_dir()
    os.environ['WANDB_SILENT'] = 'True' # suppress logs
    wandb.log(wandb_log)
    wandb.finish()
to_wandb = python_app(
        _to_wandb,
        executors=['default'],
        )
