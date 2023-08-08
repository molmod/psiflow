from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, Callable, Type, Any, NamedTuple
import typeguard
from copy import deepcopy
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import namedtuple

from ase import Atoms

import parsl
from parsl.app.app import python_app, bash_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.dataflow.memoization import id_for_memo
from parsl.executors import WorkQueueExecutor

import psiflow
from psiflow.data import Dataset, FlowAtoms, app_join_dataset, \
        app_write_dataset, NullState
from psiflow.utils import copy_data_future, unpack_i, get_active_executor, \
        copy_app_future, pack
from psiflow.walkers import BaseWalker, PlumedBias
from psiflow.models import BaseModel


Metadata = namedtuple('Metadata', ['state', 'counter', 'reset', 'temperature', 'time', 'stdout'])


#@typeguard.typechecked
def molecular_dynamics_yaff(
        device: str,
        ncores: int,
        pars: dict[str, Any],
        model_cls: str,
        keep_trajectory: bool = False,
        plumed_input: str = '',
        inputs: list[File] = [],
        outputs: list[File] = [],
        walltime: float = 1e9, # infinite by default
        stdout: str = '',
        stderr: str = '',
        parsl_resource_specification: dict = None,
        ) -> str:
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd  = 'cd $mytmpdir;'
    command_unbuffer = 'export PYTHONUNBUFFERED=TRUE;'
    command_printenv = 'printenv | grep OMP;'
    command_write = 'echo "{}" > plumed.dat;'.format(plumed_input)
    command_list = [
            command_tmp,
            command_cd,
            command_unbuffer,
            command_printenv,
            command_write,
            'timeout -k 5 {}s'.format(max(walltime - 20, 0)), # some time is spent on copying
            'psiflow-md-yaff',
            '--device {}'.format(device),
            '--ncores {}'.format(ncores),
            '--atoms {}'.format(inputs[0].filepath),
            ]
    parameters_to_pass = [
            'seed',
            'timestep',
            'steps',
            'step',
            'start',
            'temperature',
            'pressure',
            'force_threshold',
            'temperature_reset_quantile',
            ]
    for key in parameters_to_pass:
        if pars[key] is not None:
            command_list.append('--{} {}'.format(key, pars[key]))
    command_list.append('--model-cls {}'.format(model_cls))
    command_list.append('--model {}'.format(inputs[1].filepath))
    command_list.append('--keep-trajectory {}'.format(keep_trajectory))
    command_list.append('--trajectory {}'.format(outputs[0].filepath))
    command_list.append('--walltime {}'.format(walltime))
    command_list.append(' || true')
    return ' '.join(command_list)


@python_app(executors=['Default'])
def molecular_dynamics_yaff_post(
        inputs: list[File] = [],
        outputs: list[File] = [],
        ):
    from ase.io import read
    from psiflow.data import FlowAtoms, NullState
    from psiflow.walkers.utils import parse_yaff_output
    with open(inputs[1], 'r') as f:
        stdout = f.read()
    if 'unsafe' in stdout:
        reset = True
    else:
        reset = False
    counter, temperature, elapsed_time = parse_yaff_output(stdout)
    if not reset:
        atoms = FlowAtoms.from_atoms(read(str(inputs[2]))) # reads last state
    else:
        atoms = NullState
    return atoms, counter, reset, temperature, elapsed_time


#@typeguard.typechecked
def molecular_dynamics_openmm(
        device: str,
        ncores: int,
        pars: dict[str, Any],
        model_cls: str,
        keep_trajectory: bool = False,
        plumed_input: str = '',
        inputs: list[File] = [],
        outputs: list[File] = [],
        walltime: float = 1e9, # infinite by default
        stdout: str = '',
        stderr: str = '',
        parsl_resource_specification: dict = None,
        ) -> str:
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd  = 'cd $mytmpdir;'
    command_unbuffer = 'export PYTHONUNBUFFERED=TRUE;'
    command_printenv = 'printenv | grep OMP;'
    command_write = 'echo "{}" > plumed.dat;'.format(plumed_input)
    command_list = [
            command_tmp,
            command_cd,
            command_unbuffer,
            command_printenv,
            command_write,
            'timeout --kill-after=15 {}s'.format(max(walltime - 20, 0)), # some time is spent on copying
            'psiflow-md-openmm',
            '--device {}'.format(device),
            '--ncores {}'.format(ncores),
            '--atoms {}'.format(inputs[0].filepath),
            ]
    parameters_to_pass = [
            'seed',
            'timestep',
            'steps',
            'step',
            'start',
            'temperature',
            'pressure',
            'force_threshold',
            'temperature_reset_quantile',
            ]
    for key in parameters_to_pass:
        if pars[key] is not None:
            command_list.append('--{} {}'.format(key, pars[key]))
    command_list.append('--model-cls {}'.format(model_cls))
    command_list.append('--model {}'.format(inputs[1].filepath))
    command_list.append('--keep-trajectory {}'.format(keep_trajectory))
    command_list.append('--trajectory {}'.format(outputs[0].filepath))
    command_list.append('--walltime {}'.format(walltime))
    command_list.append(' || true')
    return ' '.join(command_list)


@python_app(executors=['Default'])
def molecular_dynamics_openmm_post(
        inputs: list[File] = [],
        outputs: list[File] = [],
        ):
    from ase.io import read
    from psiflow.data import FlowAtoms, NullState
    from psiflow.walkers.utils import parse_openmm_output
    with open(inputs[1], 'r') as f:
        stdout = f.read()
    if 'unsafe' in stdout:
        reset = True
    else:
        reset = False
    counter, temperature, elapsed_time = parse_openmm_output(stdout)
    if not reset:
        atoms = FlowAtoms.from_atoms(read(str(inputs[2]))) # reads last state
    else:
        atoms = NullState
    return atoms, counter, reset, temperature, elapsed_time


@typeguard.typechecked
class DynamicWalker(BaseWalker):

    def __init__(self,
            atoms: Union[Atoms, FlowAtoms, AppFuture],
            timestep: float = 0.5,
            steps: int = 100,
            step: int = 10,
            start: int = 0,
            temperature: Optional[float] = 300,
            pressure: Optional[float] = None,
            temperature_reset_quantile: float = 0.01, # bounds at 0.005 and 0.995
            force_threshold: float = 1e6, # no threshold by default
            **kwargs,
            ) -> None:
        super().__init__(atoms, **kwargs)
        self.timestep = timestep
        self.steps = steps
        self.step = step
        self.start = start
        self.temperature = temperature
        self.pressure = pressure
        self.force_threshold = force_threshold
        self.temperature_reset_quantile = temperature_reset_quantile

    @property
    def parameters(self) -> dict[str, Any]:
        parameters = super().parameters
        parameters['timestep'] = self.timestep
        parameters['steps'] = self.steps
        parameters['step'] = self.step
        parameters['start'] = self.start
        parameters['temperature'] = self.temperature
        parameters['pressure'] = self.pressure
        parameters['force_threshold'] = self.force_threshold
        parameters['temperature_reset_quantile'] = self.temperature_reset_quantile
        return parameters

    def _propagate(self, model, keep_trajectory, file):
        assert model is not None
        name = model.__class__.__name__
        context = psiflow.context()
        try:
            app = context.apps(self.__class__, 'propagate_' + name)
        except (KeyError, AssertionError):
            self.create_apps(model_cls=model.__class__)
            app = context.apps(self.__class__, 'propagate_' + name)
        return app(
                self.state,
                self.parameters,
                model,
                keep_trajectory,
                file,
                )

    @classmethod
    def create_apps(
            cls,
            model_cls: Type[BaseModel],
            ) -> None:
        """Registers propagate app in context

        While the propagate app logically belongs to the DynamicWalker, its
        execution is purely a function of the specific model instance on which
        it is called, because walker propagation is essentially just a series
        of model evaluation calls. As such, its execution is defined by the
        model itself.

        """
        context = psiflow.context()
        evaluation, _ = context[model_cls]
        ncores   = evaluation.cores_per_worker
        device   = 'cuda' if evaluation.gpu else 'cpu'
        label    = evaluation.name()
        walltime = evaluation.max_walltime
        engine   = evaluation.simulation_engine
        if isinstance(get_active_executor(label), WorkQueueExecutor):
            resource_spec = evaluation.generate_parsl_resource_specification()
        else:
            resource_spec = {}

        if engine == 'yaff':
            app_propagate = bash_app(
                    molecular_dynamics_yaff,
                    executors=[label],
                    cache=False,
                    )
            @typeguard.typechecked
            def propagate_wrapped(
                    state: AppFuture,
                    parameters: dict[str, Any],
                    model: BaseModel = None,
                    keep_trajectory: bool = False,
                    file: Optional[File] = None,
                    ) -> tuple[NamedTuple, Optional[DataFuture]]:
                assert model is not None # model is required
                future_atoms = app_write_dataset(
                        states=None,
                        inputs=[state],
                        outputs=[psiflow.context().new_file('data_', '.xyz')],
                        ).outputs[0]
                inputs = [future_atoms, model.deploy_future]
                if keep_trajectory:
                    assert file is not None
                else:
                    file = psiflow.context().new_file('data_', '.xyz')
                outputs = [file]
                future = app_propagate(
                        device,
                        ncores,
                        parameters,
                        model.__class__.__name__, # load function
                        keep_trajectory=keep_trajectory,
                        plumed_input='',
                        inputs=inputs,
                        outputs=outputs,
                        walltime=(walltime * 60),
                        stdout=parsl.AUTO_LOGNAME, # output redirected to this file
                        stderr=parsl.AUTO_LOGNAME, # error redirected to this file
                        parsl_resource_specification=resource_spec,
                        )
                result = molecular_dynamics_yaff_post(
                        inputs=[future, future.stdout, future.outputs[0]],
                        )
                metadata_args = [unpack_i(result, i) for i in range(5)]
                metadata_args.append(future.stdout)
                metadata = Metadata(*metadata_args)
                return metadata, future.outputs[0]
        elif engine == 'openmm':
            app_propagate = bash_app(
                    molecular_dynamics_openmm,
                    executors=[label],
                    )
            @typeguard.typechecked
            def propagate_wrapped(
                    state: AppFuture,
                    parameters: dict[str, Any],
                    model: BaseModel = None,
                    keep_trajectory: bool = False,
                    file: Optional[File] = None,
                    ) -> tuple[NamedTuple, Optional[DataFuture]]:
                assert model is not None # model is required
                future_atoms = app_write_dataset(
                        states=None,
                        inputs=[state],
                        outputs=[psiflow.context().new_file('data_', '.xyz')],
                        ).outputs[0]
                inputs = [future_atoms, model.deploy_future]
                if keep_trajectory:
                    assert file is not None
                else:
                    file = psiflow.context().new_file('data_', '.xyz')
                outputs = [file]
                future = app_propagate(
                        device,
                        ncores,
                        parameters,
                        model.__class__.__name__, # load function
                        keep_trajectory=keep_trajectory,
                        plumed_input='',
                        inputs=inputs,
                        outputs=outputs,
                        walltime=(walltime * 60),
                        stdout=parsl.AUTO_LOGNAME, # output redirected to this file
                        stderr=parsl.AUTO_LOGNAME, # error redirected to this file
                        parsl_resource_specification=resource_spec,
                        )
                result = molecular_dynamics_openmm_post(
                        inputs=[future, future.stdout, future.outputs[0]],
                        )
                metadata_args = [unpack_i(result, i) for i in range(5)]
                metadata_args.append(future.stdout)
                metadata = Metadata(*metadata_args)
                return metadata, future.outputs[0]

        else:
            raise ValueError('unknown simulation engine ' + engine)
        name = model_cls.__name__
        context.register_app(cls, 'propagate_' + name, propagate_wrapped)


@typeguard.typechecked
class BiasedDynamicWalker(DynamicWalker):

    def __init__(
            self,
            atoms: Union[Atoms, FlowAtoms, AppFuture],
            bias: Optional[PlumedBias] = None,
            **kwargs) -> None:
        super().__init__(atoms, **kwargs)
        assert bias is not None
        self.bias = bias.copy()

    def save(
            self,
            path: Union[Path, str],
            require_done: bool = True,
            ) -> tuple[DataFuture, ...]:
        self.bias.save(path, require_done)
        return super().save(path, require_done)

    def reset(self, condition: Union[bool, AppFuture] = None) -> None:
        self.bias.reset(condition)
        return super().reset(condition)

    def copy(self) -> BaseWalker:
        walker = self.__class__(self.state, self.bias, **self.parameters)
        walker.state0  = copy_app_future(self.state0)
        walker.counter = copy_app_future(self.counter)
        return walker

    def _propagate(self, model, keep_trajectory, file):
        assert model is not None
        name = model.__class__.__name__
        context = psiflow.context()
        try:
            app = context.apps(self.__class__, 'propagate_' + name)
        except (KeyError, AssertionError):
            self.create_apps(model_cls=model.__class__)
            app = context.apps(self.__class__, 'propagate_' + name)
        return app(
                self.state,
                self.parameters,
                self.bias,
                model,
                keep_trajectory,
                file,
                )

    @classmethod
    def distribute(cls,
            nwalkers: int,
            data_start: Dataset,
            bias: PlumedBias,
            variable: str,
            min_value: float,
            max_value: float,
            **kwargs,
            ) -> list[BiasedDynamicWalker]:
        targets = np.linspace(min_value, max_value, num=nwalkers, endpoint=True)
        data_start = bias.extract_grid(
                data_start,
                variable,
                targets=targets,
                )
        assert data_start.length().result() == nwalkers, ('could not find '
                'states for all of the CV values: {} '.format(targets))
        walkers = super(BiasedDynamicWalker, cls).multiply(
                nwalkers,
                data_start,
                bias=bias,
                **kwargs)
        bias = walkers[0].bias
        assert variable in bias.variables
        if nwalkers > 1:
            step_value = (max_value - min_value) / (nwalkers - 1)
        else:
            step_value = 0
        for i, walker in enumerate(walkers):
            center = min_value + i * step_value
            try: # do not change kappa
                walker.bias.adjust_restraint(variable, None, center)
            except AssertionError: # no restraint present on variable
                pass
        return walkers

    @classmethod
    def create_apps(
            cls,
            model_cls: Type[BaseModel],
            ) -> None:
        context = psiflow.context()
        evaluation, _ = context[model_cls]
        ncores   = evaluation.cores_per_worker
        device   = 'cuda' if evaluation.gpu else 'cpu'
        label    = evaluation.name()
        walltime = evaluation.max_walltime
        engine   = evaluation.simulation_engine
        if isinstance(get_active_executor(label), WorkQueueExecutor):
            resource_spec = evaluation.generate_parsl_resource_specification()
        else:
            resource_spec = {}

        if engine == 'yaff':
            app_propagate = bash_app(
                    molecular_dynamics_yaff,
                    executors=[label],
                    cache=False,
                    )
            @typeguard.typechecked
            def propagate_wrapped(
                    state: Union[AppFuture, FlowAtoms],
                    parameters: dict[str, Any],
                    bias: PlumedBias,
                    model: BaseModel = None,
                    keep_trajectory: bool = False,
                    file: Optional[File] = None,
                    ) -> tuple[Metadata, Optional[DataFuture]]:
                assert model is not None # model is required
                future_atoms = app_write_dataset(
                        states=None,
                        inputs=[state],
                        outputs=[psiflow.context().new_file('data_', '.xyz')],
                        ).outputs[0]
                inputs = [future_atoms, model.deploy_future]
                if keep_trajectory:
                    assert file is not None
                else:
                    file = psiflow.context().new_file('data_', '.xyz')
                outputs = [file]
                inputs += bias.futures
                outputs += [File(f.filepath) for f in bias.futures]
                future = app_propagate(
                        device,
                        ncores,
                        parameters,
                        model.__class__.__name__, # load function
                        keep_trajectory=keep_trajectory,
                        plumed_input=bias.prepare_input(),
                        inputs=inputs,
                        outputs=outputs,
                        walltime=(walltime * 60), # 20s slack
                        stdout=parsl.AUTO_LOGNAME, # output redirected to this file
                        stderr=parsl.AUTO_LOGNAME, # error redirected to this file
                        parsl_resource_specification=resource_spec,
                        )
                if 'METAD' in bias.keys:
                    bias.data_futures['METAD'] = future.outputs[1]
                result = molecular_dynamics_yaff_post(
                        inputs = [future, future.stdout, future.outputs[0]],
                        )
                state = bias.evaluate(Dataset([unpack_i(result, 0)]), as_dataset=True)[0]
                metadata_args = [state] + [unpack_i(result, i) for i in range(1, 5)]
                metadata_args.append(future.stdout)
                metadata = Metadata(*metadata_args)
                return metadata, future.outputs[0]
        elif engine == 'openmm':
            app_propagate = bash_app(
                    molecular_dynamics_openmm,
                    executors=[label],
                    )
            @typeguard.typechecked
            def propagate_wrapped(
                    state: Union[AppFuture, FlowAtoms],
                    parameters: dict[str, Any],
                    bias: PlumedBias,
                    model: BaseModel = None,
                    keep_trajectory: bool = False,
                    file: Optional[File] = None,
                    ) -> tuple[Metadata, Optional[DataFuture]]:
                assert model is not None # model is required
                future_atoms = app_write_dataset(
                        states=None,
                        inputs=[state],
                        outputs=[psiflow.context().new_file('data_', '.xyz')],
                        ).outputs[0]
                inputs = [future_atoms, model.deploy_future]
                if keep_trajectory:
                    assert file is not None
                else:
                    file = psiflow.context().new_file('data_', '.xyz')
                outputs = [file]
                inputs += bias.futures
                outputs += [File(f.filepath) for f in bias.futures]
                future = app_propagate(
                        device,
                        ncores,
                        parameters,
                        model.__class__.__name__, # load function
                        keep_trajectory=keep_trajectory,
                        plumed_input=bias.prepare_input(),
                        inputs=inputs,
                        outputs=outputs,
                        walltime=(walltime * 60), # 20s slack
                        stdout=parsl.AUTO_LOGNAME, # output redirected to this file
                        stderr=parsl.AUTO_LOGNAME, # error redirected to this file
                        parsl_resource_specification=resource_spec,
                        )
                if 'METAD' in bias.keys:
                    bias.data_futures['METAD'] = future.outputs[1]
                result = molecular_dynamics_openmm_post(
                        inputs = [future, future.stdout, future.outputs[0]],
                        )
                state = bias.evaluate(Dataset([unpack_i(result, 0)]), as_dataset=True)[0]
                metadata_args = [state] + [unpack_i(result, i) for i in range(1, 5)]
                metadata_args.append(future.stdout)
                metadata = Metadata(*metadata_args)
                return metadata, future.outputs[0]
        else:
            raise ValueError('unknown dynamics engine ' + engine)
        name = model_cls.__name__
        context.register_app(cls, 'propagate_' + name, propagate_wrapped)
