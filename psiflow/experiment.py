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
import argparse

import parsl
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.execution import ExecutionContext, generate_parsl_config
from psiflow.models import BaseModel, load_model, NequIPModel
from psiflow.reference.base import BaseReference
from psiflow.sampling import RandomWalker, PlumedBias
from psiflow.generator import Generator, save_generators, load_generators
from psiflow.data import Dataset
from psiflow.checks import Check, load_checks, SafetyCheck
from psiflow.utils import copy_app_future, log_data_to_wandb, \
        get_psiflow_config_from_file, set_file_logger


logger = logging.getLogger(__name__) # logging per module


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
def _app_log_generators(
        generator_names: list[str],
        errors: Optional[np.ndarray] = None,
        error_labels: Optional[List[str]] = None,
        bias_labels: Optional[List[str]] = None,
        inputs: List[Union[File, np.ndarray, str, bool, int]] = [],
        ) -> List[List]:
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
    dataset = Dataset(generators[0].walker.context, states)
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
                        Dataset(g.bias.context, [g.walker.state_future]),
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
class FlowManager:

    def __init__(
            self,
            path_output: Union[Path, str],
            wandb_project: str,
            wandb_group: str,
            error_kwargs: Optional[dict[str, Any]] = None,
            error_x_axis: Optional[str] = 'index',
            ) -> None:
        self.path_output = Path(path_output)
        assert self.path_output.is_dir()
        self.wandb_project = wandb_project
        self.wandb_group   = wandb_group
        if error_kwargs is None:
            error_kwargs = {
                    'metric': 'mae',
                    'properties': ['energy', 'forces', 'stress'],
                    }
        self.error_kwargs = error_kwargs
        self.error_x_axis = error_x_axis

    def dry_run(
            self,
            model: BaseModel,
            reference: BaseReference,
            generators: Optional[list[Generator]] = None,
            random_walker: Optional[RandomWalker] = None,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list[Check]] = None,
            ) -> None:
        context = model.context
        if random_walker is None:
            assert generators is not None
            random_walker = RandomWalker( # with default parameters
                    context,
                    generators[0].walker.start_future,
                    )

        # single point evaluation
        logger.info('evaluating singlepoint')
        evaluated = reference.evaluate(random_walker.state_future)
        evaluated.result()

        # generation of small dataset
        logger.info('generating from {}'.format(random_walker.__class__.__name__))
        _generators = Generator('random', random_walker).multiply(10)
        if checks is None:
            checks = [SafetyCheck()]
        else:
            checks = checks + [SafetyCheck()]
        _generators[3].walker.tag_unsafe()
        logger.info('generating samples')
        states = []
        for g in _generators:
            states.append(g(None, reference, checks=checks))
        data = Dataset(reference.context, states)
        data_success = data.get(indices=data.success)
        logger.info('generated {} states, of which {} were successful'.format(
            data.length().result(),
            data_success.length().result(),
            ))

        # short training and deploy
        model.reset()
        logger.info('initializing model')
        if (data_train is not None) and (data_valid is not None):
            assert data_train.length.result() >= 5 # only pass nonempty data
            assert data_valid.length.result() >= 2
            model.initialize(data_train)
            model.config_future.result()
            logger.info('training model')
            model.train(data_train, data_valid)
        else:
            assert data_success.length().result() >= 10
            new_train = data_success[:5]
            new_valid = data_success[5:]
            model.reset()
            model.initialize(new_train)
            model.config_future.result()
            logger.info('training model')
            model.train(new_train, new_valid)

        # deploy and generate
        model.deploy()
        if generators is not None:
            logger.info('generating samples')
            states = []
            for g in generators:
                states.append(g(model, reference, checks))
            data = Dataset(reference.context, states)
            assert data.length().result() == len(generators)

        # log and save objects
        if generators is None:
            generators = _generators
        log = self.log_wandb(
                'dry_run',
                model,
                generators,
                data_train=new_train,
                data_valid=new_valid,
                checks=checks,
                )
        log.result()
        self.save( # test save
                name='dry_run',
                model=model,
                generators=generators,
                data_train=new_train,
                data_valid=new_valid,
                checks=checks,
                )

    def save(
            self,
            name: str,
            model: BaseModel,
            generators: list[Generator],
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            data_failed: Optional[Dataset] = None,
            checks: Optional[list[Check]] = None,
            require_done=True,
            ) -> None:
        path = self.path_output / name
        path.mkdir(parents=False, exist_ok=False) # parent should exist

        # model
        model.save(path, require_done=require_done)

        # generators
        path_generators = path / 'generators'
        path_generators.mkdir(parents=False, exist_ok=False)
        save_generators(generators, path_generators, require_done=require_done)

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
            path_checks.mkdir(parents=False, exist_ok=False)
            for check in checks: # all checks stored in same dir
                check.save(path_checks, require_done=require_done)

    def load(
            self,
            name: str,
            context: ExecutionContext,
            ) -> Tuple[
                    BaseModel,
                    list[Generator],
                    Optional[Dataset],
                    Optional[Dataset],
                    Optional[list[Check]],
                    ]:
        path = self.path_output / name
        assert path.is_dir() # needs to exist

        # model
        model = load_model(context, path)
        try:
            model.create_apps(context) # ensures apps exist in context
        except AssertionError: # apps already registered
            assert context.apps(model.__class__, 'train')

        # generators
        path_generators = path / 'generators'
        generators = load_generators(context, path_generators)

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
        else:
            checks = None
        return model, generators, data_train, data_valid, checks

    def log_wandb(
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
                dataset = Dataset(model.context, check.states)
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
        return log_data_to_wandb(
                run_name,
                self.wandb_group,
                self.wandb_project,
                error_x_axis,
                list(log_futures.keys()),
                inputs=list(log_futures.values()),
                )

    def output_exists(self, name):
        return (self.path_output / name).is_dir()

    def insert_name(self, model: BaseModel) -> None:
        # set wandb
        if isinstance(model, NequIPModel):
            model.config_raw['wandb_group'] = self.wandb_group
        else:
            logger.warning('cannot set wandb name for model {}'.format(model.__class__))
            pass


@typeguard.typechecked
def initialize(args) -> Tuple[ExecutionContext, FlowManager]:
    path_experiment = Path.cwd() / args.name
    path_internal = path_experiment / 'parsl_internal'
    path_context  = path_experiment / 'context_internal'
    path_output   = path_experiment / 'output'
    if path_experiment.is_dir():
        assert args.restart, '{} already exists but restart is {}'.format(
                path_experiment,
                args.restart,
                )
        path_log       = path_experiment / 'psiflow_restart_from_{}.log'.format(args.restart)
        path_parsl_log = path_experiment / 'parsl_restart_from_{}.log'.format(args.restart)
        assert not path_log.exists(), ('log file already exists, did you '
                'already restart from this iteration?')
        assert not path_parsl_log.exists(), ('log file already exists, did you '
                'already restart from this iteration?')
        assert path_context.is_dir()
        shutil.rmtree(str(path_context))
        path_context.mkdir()
    else:
        path_experiment.mkdir()
        path_internal.mkdir()
        path_context.mkdir()
        path_output.mkdir()
        path_log       = path_experiment / 'psiflow.log'
        path_parsl_log = path_experiment / 'parsl.log'
    set_file_logger(path_log, args.psiflow_log_level)
    logger.info('setting up psiflow experiment in {}'.format(path_experiment))
    parsl_log_level = getattr(logging, args.parsl_log_level)
    parsl.set_file_logger(str(path_parsl_log), level=parsl_log_level)
    config, definitions = get_psiflow_config_from_file(
            args.psiflow_config,
            path_internal,
            )
    context = ExecutionContext(
            config,
            definitions,
            path=path_context,
            )
    context.initialize() # loads parsl config
    flow_manager = FlowManager(
            path_output=path_experiment / 'output',
            wandb_project='psiflow',
            wandb_group=args.name,
            error_x_axis=args.main_colvar,
            )
    return context, flow_manager


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--psiflow-config',
            action='store',
            type=str,
            help='psiflow configuration file to use when initializing the context',
            )
    parser.add_argument(
            '--name',
            action='store',
            type=str,
            help='names the experiment directory and wandb runs',
            )
    parser.add_argument(
            '--restart',
            action='store',
            default=None,
            type=str,
            help='the state from which to restart; i.e. "random", "0", "1" etc',
            )
    parser.add_argument(
            '--main-colvar',
            action='store',
            default='CV',
            type=str,
            help='the name of the "main" collective variable over which the error should be plotted',
            )
    parser.add_argument(
            '--psiflow-log-level',
            action='store',
            default='DEBUG',
            type=str,
            help='log level of psiflow; one of DEBUG, INFO, WARNING, ERROR, CRITICAL',
            )
    parser.add_argument(
            '--parsl-log-level',
            action='store',
            default='INFO',
            type=str,
            help='log level of parsl; one of DEBUG, INFO, WARNING, ERROR, CRITICAL',
            )
    return parser.parse_args()
