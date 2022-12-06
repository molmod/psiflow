from typing import Optional
from dataclasses import dataclass

from parsl.app.app import python_app

from autolearn.execution import ModelExecutionDefinition
from autolearn.sampling import BaseWalker
from autolearn import Bias


def simulate(device, ncores, dtype, state, parameters, load_calculator, inputs=[]):
    import torch
    import numpy as np
    from copy import deepcopy
    import yaff
    yaff.log.set_level(yaff.log.silent)
    import molmod
    from autolearn.utils import try_manual_plumed_linking
    from autolearn.sampling.utils import ForcePartASE, DataHook, \
            create_forcefield, ForceThresholdExceededException
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
    if pars.bias is not None: # add bias if present
        try_manual_plumed_linking()
        pars.bias.stage() # create tempfile with input
        part_plumed = yaff.external.ForcePartPlumed(
                forcefield.system,
                timestep=pars.timestep * molmod.units.femtosecond,
                restart=1,
                fn=pars.bias.files['input'],
                fn_log=pars.bias.files['log'],
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
        if pars.bias is not None:
            pars.bias.close()
    except ForceThresholdExceededException as e:
        print(e)
        print('tagging sample as unsafe')
        tag = 'unsafe'
        if pars.bias is not None:
            pars.bias.close(reset=True)
    yaff.log.set_level(yaff.log.silent)

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
    bias               : Optional[Bias] = None
    seed               : int = 0 # seed for randomized initializations


class DynamicWalker(BaseWalker):
    parameters_cls = DynamicParameters

    def propagate(self, model):
        """Propagates the walker in phase space using molecular dynamics"""
        device = self.context[ModelExecutionDefinition].device
        ncores = self.context[ModelExecutionDefinition].ncores
        dtype  = self.context[ModelExecutionDefinition].dtype
        assert model.future_deploy is not None
        p_simulate = python_app(
                simulate,
                executors=[self.executor_label],
                )
        result = p_simulate( # do unpacking in separate app
                device,
                ncores,
                dtype,
                self.state,
                self.parameters,
                model.load_calculator,
                inputs=[model.future_deploy],
                )
        self.state = python_app(lambda x: x[0], executors=[self.executor_label])(result)
        self.tag   = python_app(lambda x: x[1], executors=[self.executor_label])(result)
        return self.state
