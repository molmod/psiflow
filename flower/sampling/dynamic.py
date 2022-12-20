from typing import Optional
from dataclasses import dataclass

from parsl.app.app import python_app
from parsl.data_provider.files import File

from flower.execution import ModelExecutionDefinition
from flower.utils import copy_data_future, unpack_i
from flower.sampling import BaseWalker


def simulate_model(
        device,
        ncores,
        dtype,
        state,
        parameters,
        load_calculator,
        plumed_input='',
        inputs=[],
        outputs=[],
        ):
    import torch
    import os
    import tempfile
    import numpy as np
    from copy import deepcopy
    import yaff
    yaff.log.set_level(yaff.log.silent)
    import molmod
    from flower.sampling.utils import ForcePartASE, DataHook, \
            create_forcefield, ForceThresholdExceededException
    from flower.sampling.bias import try_manual_plumed_linking
    if device == 'cpu':
        torch.set_num_threads(ncores)
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
        if len(inputs) > 1:
            assert len(outputs) == 1
            with open(inputs[1], 'r') as f:
                backup_data = f.read() # backup data
        with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
            f.write(plumed_input) # write input
        path_plumed = f.name
        tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        tmp.close()
        path_log = tmp.name # dummy log file
        part_plumed = yaff.external.ForcePartPlumed(
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
    yaff.log.set_level(yaff.log.silent)

    if len(plumed_input) > 0:
        os.unlink(path_log)
        os.unlink(path_plumed)

    # update state with last stored state if data nonempty
    if len(datahook.data) > 0:
        state.set_positions(datahook.data[-1].get_positions())
        state.set_cell(datahook.data[-1].get_cell())
    return state, tag


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


class DynamicWalker(BaseWalker):
    parameters_cls = DynamicParameters

    @classmethod
    def create_apps(cls, context):
        executor_label = context[ModelExecutionDefinition].executor_label
        device = context[ModelExecutionDefinition].device
        ncores = context[ModelExecutionDefinition].ncores
        dtype = context[ModelExecutionDefinition].dtype

        path = context.path

        app_propagate = python_app(
                simulate_model,
                executors=[executor_label],
                )
        def propagate_wrapped(state, parameters, model=None, bias=None, **kwargs):
            assert model is not None # model is required
            assert model.deploy_future is not None # has to be deployed
            inputs = [model.deploy_future]
            outputs = []
            if bias is not None:
                plumed_input = bias.prepare_input()
                inputs.append(bias.data_future)
                outputs.append(File(bias.data_future.filepath))
            else:
                plumed_input = ''
            result = app_propagate(
                    device,
                    ncores,
                    dtype,
                    state,
                    parameters,
                    model.load_calculator, # load function
                    plumed_input=plumed_input,
                    inputs=inputs,
                    outputs=outputs,
                    )
            if bias is not None: # check tag and reset hills if necessary
                bias.data_future = result.outputs[0]
            return result

        context.register_app(cls, 'propagate', propagate_wrapped)
        super(DynamicWalker, cls).create_apps(context)
