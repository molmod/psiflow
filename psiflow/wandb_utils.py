from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, Callable, Dict, Tuple, Any
import typeguard
import os
import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np

import parsl
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.models import BaseModel, NequIPModel
from psiflow.reference.base import BaseReference
from psiflow.sampling import RandomWalker, PlumedBias, BiasedDynamicWalker, \
        BaseWalker
from psiflow.data import Dataset
from psiflow.checks import Check
from psiflow.utils import copy_app_future


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
def _app_log_data(
        error_x_axis: str,
        errors: np.ndarray,
        error_labels: list[str],
        inputs: list[Union[File, np.ndarray]] = [],
        ) -> list[list]:
    from ase.data import chemical_symbols
    from ase.io import write
    from psiflow.data import read_dataset
    data = read_dataset(slice(None), inputs=[inputs[0]])
    columns = ['location', 'elements', 'natoms']
    if errors.shape[0] != len(data):
        raise AssertionError('error evaluation was not performed on every state')
    assert len(error_labels) == errors.shape[1]
    columns += error_labels
    columns.append(error_x_axis)
    table_data = []
    location = Path(inputs[0].filepath).name
    for i, atoms in enumerate(data):
        elements = list(set([chemical_symbols[n] for n in atoms.numbers]))
        row = [location, ', '.join(elements), len(atoms)]
        row += [e for e in errors[i, :]]
        if error_x_axis != 'index':
            assert error_x_axis in atoms.info
            row.append(atoms.info[error_x_axis])
        else:
            row.append(i)
        assert len(columns) == len(row)
        table_data.append(row)
    return [columns] + table_data
app_log_data = python_app(_app_log_data, executors=['default'])


@typeguard.typechecked
def log_data(
        dataset: Dataset,
        model: BaseModel,
        error_x_axis: str,
        error_kwargs: dict[str, dict],
        ) -> dict[str, AppFuture]:
    if len(model.deploy_future) == 0:
        model.deploy()
    evaluated = model.evaluate(dataset)
    log = {}
    for suffix, kwargs_dict in error_kwargs.items():
        errors = Dataset.get_errors(
                dataset,
                evaluated,
                **kwargs_dict,
                )
        error_labels = [kwargs_dict['metric'] + '_' + p for p in kwargs_dict['properties']]
        log[suffix] = app_log_data(
                error_x_axis,
                errors=errors,
                error_labels=error_labels,
                inputs=[dataset.data_future],
                )
    return log


@typeguard.typechecked
@dataclass
class WandBLogger:
    wandb_project: str
    wandb_group: str
    error_x_axis: str = 'index'
    metric: str = 'mae'
    elements: Optional[list[str]] = None
    indices: Optional[list[int]] = None

    def __call__(
            self,
            run_name: str,
            model: BaseModel,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            data_failed: Optional[Dataset] = None,
            ) -> AppFuture:
        logger.info('logging data to wandb')
        x_axis_present = True
        if data_train is not None:
            x_axis_present *= self.error_x_axis in data_train.info_keys().result()
        if data_valid is not None:
            x_axis_present *= self.error_x_axis in data_valid.info_keys().result()
        if data_failed is not None:
            x_axis_present *= self.error_x_axis in data_failed.info_keys().result()
        if x_axis_present:
            error_x_axis = self.error_x_axis
        else:
            logger.critical('could not find variable "{}" in XYZ header of data'
                    '; falling back to using state index during logging'.format(
                        self.error_x_axis))
            error_x_axis = 'index'

        # build error_kwargs
        error_kwargs = {'all': {
                    'metric': self.metric,
                    'properties': ['energy', 'forces', 'stress']}
                }
        if self.elements is not None:
            for element in self.elements:
                error_kwargs[element] = {
                        'metric': self.metric,
                        'properties': ['forces'],
                        'elements': [element],
                        }
        if self.indices is not None:
            for index in self.indices:
                error_kwargs['index' + str(index)] = {
                        'metric': self.metric,
                        'properties': ['forces'],
                        'atom_indices': [index],
                        }

        log_futures = {}
        if data_train is not None:
            log = log_data(data_train, model, error_x_axis, error_kwargs)
            for suffix, value in log.items():
                log_futures['training_' + suffix] = value
        if data_valid is not None:
            log = log_data(data_valid, model, error_x_axis, error_kwargs)
            for suffix, value in log.items():
                log_futures['validation_' + suffix] = value
        if data_failed is not None:
            log = log_data(data_failed, model, error_x_axis, error_kwargs)
            for suffix, value in log.items():
                log_futures['failed_' + suffix] = value
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
                list(log_futures.keys()),
                inputs=list(log_futures.values()),
                )

    def insert_name(self, model: BaseModel) -> None:
        if isinstance(model, NequIPModel):
            model.config_raw['wandb_group'] = self.wandb_group
        else:
            logger.warning('cannot set wandb name for model {}'.format(model.__class__))


@typeguard.typechecked
def _to_wandb(
        run_name: str,
        group: str,
        project: str,
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
        if any(name.find(key) for key in ['training', 'validation', 'failed']):
            errors_to_plot = [] # check which error labels are present
            for l in data[0]:
                if ('energy' in l) or ('forces' in l) or ('stress' in l):
                    errors_to_plot.append(l)
            for error in errors_to_plot:
                data_name = name.split('_')[0]
                suffix = name.split('_')[1]
                metric = error.split('_')[0]
                property_ = error.split('_')[1]
                title = (property_ + '_' + data_name) + '/' + metric + '_' + suffix
                wandb_log[title] = wandb.plot.scatter(
                        table,
                        data[0][-1], # index or CV
                        error,
                        title=title,
                        )
        else:
            wandb_log['tables/' + name] = table
    assert path_wandb.is_dir()
    os.environ['WANDB_SILENT'] = 'True' # suppress logs
    wandb.log(wandb_log)
    wandb.finish()
to_wandb = python_app(_to_wandb, executors=['default'])
