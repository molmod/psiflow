from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Callable, Tuple, Type
import typeguard
from dataclasses import dataclass, asdict

import parsl
from parsl.app.app import python_app, bash_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.dataflow.memoization import id_for_memo
from parsl.executors import WorkQueueExecutor

from psiflow.data import Dataset, FlowAtoms
from psiflow.execution import ExecutionContext, ModelEvaluationExecution
from psiflow.utils import copy_data_future, unpack_i, get_active_executor
from psiflow.sampling import BaseWalker, PlumedBias
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
        inputs: List[File] =[],
        outputs: List[File] = [],
        walltime: float = 1e12, # infinite by default
        stdout: str = '',
        #stderr: str = '',
        parsl_resource_specification: dict = None,
        ) -> Tuple[FlowAtoms, str]:
    import torch
    import os
    import tempfile
    import numpy as np
    import parsl
    import logging
    from copy import deepcopy
    from parsl.utils import get_std_fname_mode # from parsl/app/bash.py
    fname, mode = get_std_fname_mode('stdout', stdout)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fd = open(fname, mode)
    import yaff
    yaff.log.set_file(fd)
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
    atoms = state.copy()
    atoms.calc = load_calculator(inputs[0].filepath, device, dtype)
    forcefield = create_forcefield(atoms, pars.force_threshold)

    loghook  = yaff.VerletScreenLog(step=pars.step, start=0)
    datahook = DataHook(start=pars.start, step=pars.step)
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
    try: # exception may already be raised at initialization of verlet
        verlet = yaff.VerletIntegrator(
                forcefield,
                timestep=pars.timestep*molmod.units.femtosecond,
                hooks=hooks,
                temp0=pars.initial_temperature,
                )
        yaff.log.set_level(yaff.log.medium)
        verlet.run(pars.steps)
    except ForceThresholdExceededException as e:
        print(e)
        print('tagging sample as unsafe')
        tag = 'unsafe'
        if len(plumed_input) > 0:
            if len(inputs) > 1:
                with open(inputs[1], 'w') as f: # reset hills
                    f.write(backup_data)
    except parsl.app.errors.AppTimeout as e:
        print(e)
    yaff.log.set_level(yaff.log.silent)
    fd.close()

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
    return FlowAtoms.from_atoms(state), tag


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


@id_for_memo.register(DynamicParameters)
def id_for_memo_cp2k_parameters(parameters: DynamicParameters, output_ref=False):
    assert not output_ref
    return id_for_memo(asdict(parameters), output_ref=output_ref)


@typeguard.typechecked
class DynamicWalker(BaseWalker):
    parameters_cls = DynamicParameters

    def get_propagate_app(self, model):
        name = model.__class__.__name__
        try:
            app = self.context.apps(DynamicWalker, 'propagate_' + name)
        except (KeyError, AssertionError):
            assert model.__class__ in self.context.definitions.keys()
            self.create_apps(self.context, model_cls=model.__class__)
            app = self.context.apps(DynamicWalker, 'propagate_' + name)
        return app

    @classmethod
    def create_apps(
            cls,
            context: ExecutionContext,
            model_cls: Type[BaseModel],
            ) -> None:
        """Registers propagate app in context

        While the propagate app logically belongs to the DynamicWalker, its
        execution is purely a function of the specific model instance on which
        it is called, because walker propagation is essentially just a series
        of model evaluation calls. As such, its execution is defined by the
        model itself.

        """
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
                bias: Optional[PlumedBias] = None,
                keep_trajectory: bool = False,
                file: Optional[File] = None,
                **kwargs,
                ) -> AppFuture:
            assert model is not None # model is required
            assert model.deploy_future[dtype] is not None # has to be deployed
            inputs = [model.deploy_future[dtype]]
            outputs = []
            if keep_trajectory:
                assert file is not None
                outputs.append(file)
            if bias is not None:
                plumed_input = bias.prepare_input()
                inputs += list(bias.data_futures.values())
                outputs += [File(f.filepath) for f in bias.data_futures.values()]
            else:
                plumed_input = ''
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
                    stdout=parsl.AUTO_LOGNAME,
                    parsl_resource_specification=resource_spec,
                    )
            if bias is not None: # ensure dependency on new hills is set
                if 'METAD' in bias.keys:
                    if keep_trajectory:
                        index = 1
                    else:
                        index = 0
                    bias.data_futures['METAD'] = result.outputs[index]
            return result
        # register using model_cls, not cls!
        name = model_cls.__name__
        context.register_app(cls, 'propagate_' + name, propagate_wrapped)
        super(DynamicWalker, cls).create_apps(context)
