from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, Callable, Type, Any
import typeguard
from copy import deepcopy
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

from ase import Atoms

import parsl
from parsl.app.app import python_app, bash_app, join_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.dataflow.memoization import id_for_memo
from parsl.executors import WorkQueueExecutor

import psiflow
from psiflow.data import Dataset, FlowAtoms, app_join_dataset, \
        app_save_dataset
from psiflow.execution import ModelEvaluationExecution
from psiflow.utils import copy_data_future, unpack_i, get_active_executor, \
        copy_app_future, pack
from psiflow.sampling import BaseWalker, PlumedBias
from psiflow.sampling.utils import parse_yaff_output
from psiflow.sampling.base import sum_counters, update_tag, conditional_reset
from psiflow.models import BaseModel


#@typeguard.typechecked
def molecular_dynamics_yaff(
        device: str,
        ncores: int,
        dtype: str,
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
    command_write = 'echo "{}" > plumed.dat;'.format(plumed_input)
    command_list = [
            command_tmp,
            command_cd,
            command_unbuffer,
            command_write,
            'timeout -k 5 {}s'.format(max(walltime - 100, 0)), # some time is spent on copying
            'psiflow-md-yaff',
            '--device {}'.format(device),
            '--ncores {}'.format(ncores),
            '--dtype {}'.format(dtype),
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
            'initial_temperature',
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


def molecular_dynamics_yaff_post(
        inputs: list[File] = [],
        outputs: list[File] = [],
        ):
    from ase.io import read
    from psiflow.data import FlowAtoms
    with open(inputs[1], 'r') as f:
        stdout = f.read()
    tag, counter = parse_yaff_output(stdout)
    atoms = FlowAtoms.from_atoms(read(str(inputs[2]))) # reads last state
    return atoms, tag, counter


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
            force_threshold: float = 1e6, # no threshold by default
            initial_temperature: float = 600, # to mimick parallel tempering
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
        self.initial_temperature = initial_temperature

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
        parameters['initial_temperature'] = self.initial_temperature
        return parameters

    def _propagate(self, model, keep_trajectory, file):
        assert model is not None
        name = model.__class__.__name__
        context = psiflow.context()
        try:
            app = context.apps(self.__class__, 'propagate_' + name)
        except (KeyError, AssertionError):
            assert model.__class__ in context.definitions.keys()
            self.create_apps(model_cls=model.__class__)
            app = context.apps(self.__class__, 'propagate_' + name)
        return app(
                self.state_future,
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
        for execution in context[model_cls]:
            if type(execution) == ModelEvaluationExecution:
                label    = execution.executor
                device   = execution.device
                dtype    = execution.dtype
                ncores   = execution.ncores
                walltime = execution.walltime
                if isinstance(get_active_executor(label), WorkQueueExecutor):
                    resource_spec = execution.generate_parsl_resource_specification()
                else:
                    resource_spec = {}
        if walltime is None:
            walltime = 1e4 # infinite

        app_propagate = bash_app(
                molecular_dynamics_yaff,
                executors=[label],
                cache=False,
                )
        app_propagate_post = python_app(
                molecular_dynamics_yaff_post,
                executors=['default'],
                cache=False,
                )

        @typeguard.typechecked
        def propagate_wrapped(
                state: AppFuture,
                parameters: dict[str, Any],
                model: BaseModel = None,
                keep_trajectory: bool = False,
                file: Optional[File] = None,
                ) -> tuple[AppFuture, Optional[DataFuture]]:
            assert model is not None # model is required
            assert model.deploy_future[dtype] is not None # has to be deployed
            future_atoms = app_save_dataset(
                    states=None,
                    inputs=[state],
                    outputs=[psiflow.context().new_file('data_', '.xyz')],
                    ).outputs[0]
            inputs = [future_atoms, model.deploy_future[dtype]]
            if keep_trajectory:
                assert file is not None
            else:
                file = psiflow.context().new_file('data_', '.xyz')
            outputs = [file]
            future = app_propagate(
                    device,
                    ncores,
                    dtype,
                    parameters,
                    model.__class__.__name__, # load function
                    keep_trajectory=keep_trajectory,
                    plumed_input='',
                    inputs=inputs,
                    outputs=outputs,
                    walltime=(walltime * 60 - 20), # 20s slack
                    stdout=parsl.AUTO_LOGNAME, # output redirected to this file
                    stderr=parsl.AUTO_LOGNAME, # error redirected to this file
                    parsl_resource_specification=resource_spec,
                    )
            result = app_propagate_post(
                    inputs=[future, future.stdout, future.outputs[0]],
                    )
            return result, future.outputs[0]
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

    def reset(self, conditional: bool = False) -> AppFuture:
        return super().reset(conditional)

    def copy(self) -> BaseWalker:
        walker = self.__class__(self.state_future, self.bias, **self.parameters)
        walker.start_future = copy_app_future(self.start_future)
        walker.tag_future   = copy_app_future(self.tag_future)
        return walker

    def _propagate(self, model, keep_trajectory, file):
        assert model is not None
        name = model.__class__.__name__
        context = psiflow.context()
        try:
            app = context.apps(self.__class__, 'propagate_' + name)
        except (KeyError, AssertionError):
            assert model.__class__ in context.definitions.keys()
            self.create_apps(model_cls=model.__class__)
            app = context.apps(self.__class__, 'propagate_' + name)
        return app(
                self.state_future,
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
        for execution in context[model_cls]:
            if type(execution) == ModelEvaluationExecution:
                label    = execution.executor
                device   = execution.device
                dtype    = execution.dtype
                ncores   = execution.ncores
                walltime = execution.walltime
                if isinstance(get_active_executor(label), WorkQueueExecutor):
                    resource_spec = execution.generate_parsl_resource_specification()
                else:
                    resource_spec = {}
        if walltime is None:
            walltime = 1e4 # infinite

        app_propagate = bash_app(
                molecular_dynamics_yaff,
                executors=[label],
                cache=False,
                )
        app_propagate_post = python_app(
                molecular_dynamics_yaff_post,
                executors=['default'],
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
                ) -> tuple[AppFuture, Optional[DataFuture]]:
            assert model is not None # model is required
            assert model.deploy_future[dtype] is not None # has to be deployed
            future_atoms = app_save_dataset(
                    states=None,
                    inputs=[state],
                    outputs=[psiflow.context().new_file('data_', '.xyz')],
                    ).outputs[0]
            inputs = [future_atoms, model.deploy_future[dtype]]
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
                    dtype,
                    parameters,
                    model.__class__.__name__, # load function
                    keep_trajectory=keep_trajectory,
                    plumed_input=bias.prepare_input(),
                    inputs=inputs,
                    outputs=outputs,
                    walltime=(walltime * 60 - 20), # 20s slack
                    stdout=parsl.AUTO_LOGNAME, # output redirected to this file
                    stderr=parsl.AUTO_LOGNAME, # error redirected to this file
                    parsl_resource_specification=resource_spec,
                    )
            if 'METAD' in bias.keys:
                bias.data_futures['METAD'] = future.outputs[1]
            result = app_propagate_post(
                    inputs = [future, future.stdout, future.outputs[0]],
                    )
            return result, future.outputs[0]
        name = model_cls.__name__
        context.register_app(cls, 'propagate_' + name, propagate_wrapped)


@python_app
def determine_center(
        steps: int,
        counter: int,
        targets: np.ndarray,
        ) -> float:
    index = counter // steps
    i = index % (2 * len(targets))
    if i >= len(targets):
        i = len(targets) - i % len(targets) - 1
    assert i >= 0
    assert i < len(targets)
    return targets[i]


@join_app
def propagate_single(
        state: FlowAtoms,
        walker: MovingRestraintDynamicWalker,
        model: BaseModel,
        center: float,
        keep_trajectory: bool,
        file,
        ) -> Any:
    walker.bias.adjust_restraint(
            walker.variable,
            kappa=None,
            center=center,
            )
    assert model is not None
    name = model.__class__.__name__
    context = psiflow.context()
    try:
        app = context.apps(walker.__class__, 'propagate_' + name)
    except (KeyError, AssertionError):
        assert model.__class__ in context.definitions.keys()
        walker.create_apps(model_cls=model.__class__)
        app = context.apps(walker.__class__, 'propagate_' + name)
    result, output = app(
            state,
            walker.parameters,
            walker.bias,
            model,
            keep_trajectory,
            file,
            )
    #result.outputs.append(output)
    return pack(result, output)


@join_app
def combine_output(files, file, *args):
    return 


class MovingRestraintDynamicWalker(BiasedDynamicWalker):

    def __init__(
            self,
            atoms: Union[Atoms, FlowAtoms, AppFuture],
            variable: str = None,
            min_value: float = None,
            max_value: float = None,
            increment: float = None,
            num_propagations: int = None,
            **kwargs) -> None:
        super().__init__(atoms, **kwargs)
        assert variable in self.bias.variables
        self.variable = variable
        self.min_value = min_value
        self.max_value = max_value
        self.increment = increment
        self.num_propagations = num_propagations
        self.targets = np.linspace(
                min_value,
                max_value,
                int((max_value - min_value) / increment + 1),
                endpoint=True,
                )

    def _propagate(self, model, keep_trajectory, file):
        # if targets are [a, b, c]; then i will select according to
        # counter 0         : a
        # counter 1 * steps : b
        # counter 2 * steps : c
        # counter 3 * steps : b
        # counter 4 * steps : a
        # counter 5 * steps : b etc
        files = []
        for j in range(self.num_propagations):
            if keep_trajectory:
                _file = psiflow.context().new_file('data_', '.xyz')
            else:
                _file = None
            center = determine_center(
                    self.steps,
                    self.counter_future,
                    self.targets,
                    )
            join_result = propagate_single(
                    self.state_future, # pass starting state explicitly!
                    self,
                    model,
                    center,
                    keep_trajectory,
                    _file,
                    )
            future = unpack_i(join_result, 0)
            output = unpack_i(join_result, 1)
            self.state_future   = unpack_i(future, 0)
            self.tag_future     = update_tag(self.tag_future, unpack_i(future, 1))
            self.counter_future = sum_counters(self.counter_future, unpack_i(future, 2))
            self.reset(conditional=True)
            files.append(output)
        result = pack(self.state_future, self.tag_future, self.counter_future)
        self.counter_future = 0 # hack to avoid double counting in propagate()!!
        if file is not None:
            output = app_join_dataset(inputs=files, outputs=[file]).outputs[0]
        else:
            output = None
        return result, output

    @property
    def parameters(self) -> dict[str, Any]:
        parameters = super().parameters
        parameters['variable'] = self.variable
        parameters['min_value'] = self.min_value
        parameters['max_value'] = self.max_value
        parameters['increment'] = self.increment
        parameters['num_propagations'] = self.num_propagations
        return parameters

    @classmethod
    def distribute(cls, *args, **kwargs):
        raise NotImplementedError
