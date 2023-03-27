from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Callable, Tuple, Type, Any
import typeguard
from copy import deepcopy
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

import parsl
from parsl.app.app import python_app, bash_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.dataflow.memoization import id_for_memo
from parsl.executors import WorkQueueExecutor

from ase import Atoms

import psiflow
from psiflow.data import Dataset, FlowAtoms, app_join_dataset
from psiflow.execution import ModelEvaluationExecution
from psiflow.utils import copy_data_future, unpack_i, get_active_executor, \
        copy_app_future
from psiflow.sampling import BaseWalker, PlumedBias
from psiflow.sampling.base import sum_counters, update_tag, conditional_reset
from psiflow.models import BaseModel


#@typeguard.typechecked
def simulate_model(
        device: str,
        ncores: int,
        dtype: str,
        state: FlowAtoms,
        parameters: DynamicParameters,
        load_calculator: Callable,
        keep_trajectory: bool = False,
        plumed_input: str = '',
        inputs: List[File] = [],
        outputs: List[File] = [],
        walltime: float = 1e12, # infinite by default
        stdout: str = '',
        stderr: str = '',
        #stderr: str = '',
        parsl_resource_specification: dict = None,
        ) -> Tuple[FlowAtoms, str, int]:
    import torch
    import os
    import tempfile
    import numpy as np
    import parsl
    import logging
    from copy import deepcopy

    # capture output and error
    from parsl.utils import get_std_fname_mode # from parsl/app/bash.py
    from contextlib import redirect_stderr
    fname, mode = get_std_fname_mode('stderr', stderr)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fde = open(fname, mode)
    with redirect_stderr(fde): # redirect stderr
        fname, mode = get_std_fname_mode('stdout', stdout)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fdo = open(fname, mode)
        import yaff
        yaff.log.set_file(fdo) # redirect yaff log
        import molmod
        from ase.io.extxyz import write_extxyz
        from psiflow.sampling.utils import ForcePartASE, DataHook, \
                create_forcefield, ForceThresholdExceededException, ForcePartPlumed
        from psiflow.sampling.bias import try_manual_plumed_linking
        if device == 'cpu':
            torch.set_num_threads(ncores)
        if dtype == 'float64':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
        pars = parameters
        np.random.seed(pars.seed)
        torch.manual_seed(pars.seed)
        atoms   = state.copy()
        initial = state.copy()
        atoms.calc = load_calculator(inputs[0].filepath, device, dtype)
        forcefield = create_forcefield(atoms, pars.force_threshold)

        loghook  = yaff.VerletScreenLog(step=pars.step, start=0)
        datahook = DataHook() # bug in YAFF: override start/step after init
        datahook.start = pars.start
        datahook.step  = pars.step
        hooks = []
        hooks.append(loghook)
        hooks.append(datahook)
        if len(plumed_input) > 0: # add bias if present
            try_manual_plumed_linking()
            if len(inputs) > 1: # item 1 is hills file; only one to backup
                with open(inputs[1], 'r') as f: # always exists
                    backup_data = f.read() # backup data
            with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
                f.write(plumed_input) # write input
            path_plumed = f.name
            tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
            tmp.close()
            path_log = tmp.name # dummy log file
            part_plumed = ForcePartPlumed(
                    forcefield.system,
                    timestep=pars.timestep * molmod.units.femtosecond,
                    restart=1,
                    fn=path_plumed,
                    fn_log=path_log,
                    )
            forcefield.add_part(part_plumed)
            hooks.append(part_plumed) # NECESSARY!!

        thermo = yaff.LangevinThermostat(
                pars.temperature,
                timecon=100 * molmod.units.femtosecond,
                )
        if pars.pressure is None:
            print('sampling NVT ensemble ...')
            hooks.append(thermo)
        else:
            print('sampling NPT ensemble ...')
            try: # some models do not have stress support; prevent NPT!
                stress = atoms.get_stress()
            except Exception as e:
                raise ValueError('NPT requires stress support in model')
            baro = yaff.LangevinBarostat(
                    forcefield,
                    pars.temperature,
                    pars.pressure * 1e6 * molmod.units.pascal, # in MPa
                    timecon=molmod.units.picosecond,
                    anisotropic=True,
                    vol_constraint=False,
                    )
            tbc = yaff.TBCombination(thermo, baro)
            hooks.append(tbc)

        tag = 'safe'
        counter = 0
        try: # exception may already be raised at initialization of verlet
            verlet = yaff.VerletIntegrator(
                    forcefield,
                    timestep=pars.timestep*molmod.units.femtosecond,
                    hooks=hooks,
                    temp0=pars.initial_temperature,
                    )
            yaff.log.set_level(yaff.log.medium)
            verlet.run(pars.steps)
            counter = verlet.counter
        except ForceThresholdExceededException as e:
            print(e)
            print('tagging sample as unsafe')
            tag = 'unsafe'
            try:
                counter = verlet.counter
            except UnboundLocalError: # if it happened during verlet init
                pass
        except parsl.app.errors.AppTimeout as e:
            counter = verlet.counter
            print(e)
        yaff.log.set_level(yaff.log.silent)
        fdo.close()

        if len(plumed_input) > 0:
            os.unlink(path_log)
            os.unlink(path_plumed)

        # update state with last stored state if data nonempty
        if len(datahook.data) > 0:
            state.set_positions(datahook.data[-1].get_positions())
            state.set_cell(datahook.data[-1].get_cell())

        # write data to output xyz
        if keep_trajectory:
            assert str(outputs[0].filepath).endswith('.xyz')
            with open(outputs[0], 'w+') as f:
                write_extxyz(f, datahook.data)

        # check whether counter == 0 actually means state = start
        counter_is_reset = counter == 0
        state_is_reset   = np.allclose(
                    initial.get_positions(),
                    state.get_positions(),
                    )
        if state_is_reset:
            if len(plumed_input) > 0:
                if len(inputs) > 1:
                    with open(inputs[1], 'w') as f: # reset hills
                        f.write(backup_data)
        if counter_is_reset: assert state_is_reset
        if state_is_reset and (pars.step == 1): assert counter_is_reset
    return FlowAtoms.from_atoms(state), tag, counter


@typeguard.typechecked
@dataclass
class DynamicParameters: # container dataclass for simulation parameters
    timestep           : float = 0.5
    steps              : int = 100
    step               : int = 10
    start              : int = 0
    temperature        : float = 300
    pressure           : Optional[float] = None
    force_threshold    : float = 1e6 # no threshold by default
    initial_temperature: float = 600 # to mimick parallel tempering
    seed               : int = 0 # seed for randomized initializations


@typeguard.typechecked
class DynamicWalker(BaseWalker):
    parameters_cls = DynamicParameters

    def get_propagate_app(self, model):
        name = model.__class__.__name__
        context = psiflow.context()
        try:
            app = context.apps(self.__class__, 'propagate_' + name)
        except (KeyError, AssertionError):
            assert model.__class__ in context.definitions.keys()
            self.create_apps(model_cls=model.__class__)
            app = context.apps(self.__class__, 'propagate_' + name)
        return app

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
            walltime = 1e10 # infinite

        app_propagate = python_app(
                simulate_model,
                executors=[label],
                cache=False,
                )

        @typeguard.typechecked
        def propagate_wrapped(
                state: AppFuture,
                parameters: DynamicParameters,
                model: BaseModel = None,
                keep_trajectory: bool = False,
                file: Optional[File] = None,
                ) -> AppFuture:
            assert model is not None # model is required
            assert model.deploy_future[dtype] is not None # has to be deployed
            inputs = [model.deploy_future[dtype]]
            outputs = []
            if keep_trajectory:
                assert file is not None
                outputs.append(file)
            result = app_propagate(
                    device,
                    ncores,
                    dtype,
                    state,
                    parameters,
                    model.load_calculator, # load function
                    keep_trajectory=keep_trajectory,
                    plumed_input='',
                    inputs=inputs,
                    outputs=outputs,
                    walltime=(walltime * 60 - 20), # 20s slack
                    stdout=parsl.AUTO_LOGNAME, # output redirected to this file
                    stderr=parsl.AUTO_LOGNAME, # error redirected to this file
                    parsl_resource_specification=resource_spec,
                    )
            return result
        name = model_cls.__name__
        context.register_app(cls, 'propagate_' + name, propagate_wrapped)


@typeguard.typechecked
class BiasedDynamicWalker(DynamicWalker):

    def __init__(
            self,
            atoms: Union[Atoms, FlowAtoms, AppFuture],
            bias: Optional[PlumedBias] = None,
            **kwargs) -> None:
        assert bias is not None
        self.bias = bias.copy()
        super().__init__(atoms, **kwargs)

    def copy(self):
        walker = self.__class__(self.state_future, self.bias.copy())
        walker.start_future = copy_app_future(self.start_future)
        walker.tag_future   = copy_app_future(self.tag_future)
        walker.parameters   = deepcopy(self.parameters)
        return walker

    def save(
            self,
            path: Union[Path, str],
            require_done: bool = True,
            ) -> tuple[DataFuture, ...]:
        self.bias.save(path, require_done)
        return super().save(path, require_done)

    def propagate(
            self,
            safe_return: bool = False,
            keep_trajectory: bool = False,
            model: Optional[BaseModel] = None,
            ) -> Union[AppFuture, Tuple[AppFuture, Dataset]]:
        app = self.get_propagate_app(model)
        if keep_trajectory:
            file = psiflow.context().new_file('data_', '.xyz')
        else:
            file = None
        result = app(
                self.state_future,
                deepcopy(self.parameters),
                self.bias,
                model=model,
                keep_trajectory=keep_trajectory,
                file=file,
                )
        self.state_future   = unpack_i(result, 0)
        self.tag_future     = update_tag(self.tag_future, unpack_i(result, 1))
        self.counter_future = sum_counters(self.counter_future, unpack_i(result, 2))
        if safe_return: # only return state if safe, else return start
            # this does NOT reset the walker!
            _ = conditional_reset(
                    self.state_future,
                    self.start_future,
                    self.tag_future,
                    self.counter_future,
                    conditional=True
                    )
            future = unpack_i(_, 0)
        else:
            future = self.state_future
        future = copy_app_future(future) # necessary
        if keep_trajectory:
            return future, Dataset(None, data_future=result.outputs[0])
        else:
            return future

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
            walltime = 1e10 # infinite

        app_propagate = python_app(
                simulate_model,
                executors=[label],
                cache=False,
                )

        @typeguard.typechecked
        def propagate_wrapped(
                state: AppFuture,
                parameters: Any,
                bias: PlumedBias,
                model: BaseModel = None,
                keep_trajectory: bool = False,
                file: Optional[File] = None,
                ) -> AppFuture:
            assert model is not None # model is required
            assert model.deploy_future[dtype] is not None # has to be deployed
            inputs = [model.deploy_future[dtype]]
            outputs = []
            if keep_trajectory:
                assert file is not None
                outputs.append(file)
            plumed_input = bias.prepare_input()
            inputs += bias.futures
            outputs += [File(f.filepath) for f in bias.futures]
            result = app_propagate(
                    device,
                    ncores,
                    dtype,
                    state,
                    parameters,
                    model.load_calculator, # load function
                    keep_trajectory=keep_trajectory,
                    plumed_input=plumed_input,
                    inputs=inputs,
                    outputs=outputs,
                    walltime=(walltime * 60 - 20), # 20s slack
                    stdout=parsl.AUTO_LOGNAME, # output redirected to this file
                    stderr=parsl.AUTO_LOGNAME, # error redirected to this file
                    parsl_resource_specification=resource_spec,
                    )
            if 'METAD' in bias.keys:
                if keep_trajectory:
                    index = 1
                else:
                    index = 0
                bias.data_futures['METAD'] = result.outputs[index]
            return result
        name = model_cls.__name__
        context.register_app(cls, 'propagate_' + name, propagate_wrapped)


@typeguard.typechecked
@dataclass
class MovingRestraintDynamicParameters:
    variable           : str    # mandatory args
    min_value          : float
    max_value          : float
    increment          : float
    num_propagations   : int
    index              : int = 0
    timestep           : float = 0.5
    steps              : int = 100
    step               : int = 10
    start              : int = 0
    temperature        : float = 300
    pressure           : Optional[float] = None
    force_threshold    : float = 1e6 # no threshold by default
    initial_temperature: float = 600 # to mimick parallel tempering
    seed               : int = 0 # seed for randomized initializations


class MovingRestraintDynamicWalker(BiasedDynamicWalker):
    parameters_cls = MovingRestraintDynamicParameters

    def __init__(
            self,
            atoms: Union[Atoms, FlowAtoms, AppFuture],
            **kwargs) -> None:
        super().__init__(atoms, **kwargs)
        assert self.parameters.variable in self.bias.variables
        self.targets  = np.linspace(
                self.parameters.min_value,
                self.parameters.max_value,
                int((self.parameters.max_value - self.parameters.min_value) / self.parameters.increment + 1),
                endpoint=True,
                )

    def propagate(
            self,
            safe_return: bool = False,
            keep_trajectory: bool = False,
            model: Optional[BaseModel] = None,
            ) -> Union[AppFuture, Tuple[AppFuture, Dataset]]:
        app = self.get_propagate_app(model)

        # if targets are [a, b, c]; then i will select according to
        # index 0: a
        # index 1: b
        # index 2: c
        # index 3: b
        # index 4: a
        # index 5: b etc
        files = []
        for j in range(self.parameters.num_propagations):
            if keep_trajectory:
                file = psiflow.context().new_file('data_', '.xyz')
            else:
                file = None
            i = self.parameters.index % (2 * len(self.targets))
            if i >= len(self.targets):
                i = len(self.targets) - i % len(self.targets) - 1
            assert i >= 0
            assert i < len(self.targets)
            self.bias.adjust_restraint(
                    self.parameters.variable,
                    kappa=None,
                    center=self.targets[i],
                    )
            self.parameters.index += 1
            result = app(
                    self.state_future,
                    deepcopy(self.parameters),
                    self.bias,
                    model=model,
                    keep_trajectory=keep_trajectory,
                    file=file,
                    )
            if keep_trajectory:
                files.append(result.outputs[0])
            self.state_future   = unpack_i(result, 0)
            self.tag_future     = update_tag(self.tag_future, unpack_i(result, 1))
            self.counter_future = sum_counters(self.counter_future, unpack_i(result, 2))
        if safe_return: # only return state if safe, else return start
            # this does NOT reset the walker!
            _ = conditional_reset(
                    self.state_future,
                    self.start_future,
                    self.tag_future,
                    self.counter_future,
                    conditional=True
                    )
            future = unpack_i(_, 0)
        else:
            future = self.state_future
        future = copy_app_future(future) # necessary
        if keep_trajectory:
            join_future = app_join_dataset(inputs=files, outputs=[psiflow.context().new_file('data_', '.xyz')])
            return future, Dataset(None, data_future=join_future.outputs[0])
        else:
            return future

    @classmethod
    def distribute(cls, *args, **kwargs):
        raise NotImplementedError
