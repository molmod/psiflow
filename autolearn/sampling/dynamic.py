import os
import tempfile
from typing import Optional
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import torch
import molmod
import yaff
import covalent as ct

from autolearn.base import BaseWalker
from .utils import ForcePartASE, DataHook, create_forcefield, \
        ForceThresholdExceededException, try_manual_plumed_linking


yaff.log.set_level(yaff.log.silent)


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
    plumed_input       : Optional[str] = None # optional bias configuration
    seed               : int = 0 # seed for randomized initializations


class DynamicWalker(BaseWalker):

    def __init__(self, sample, **kwargs):
        start = deepcopy(sample)
        start.clear()
        self.start = start
        self.state = deepcopy(start)
        self.parameters = DynamicParameters(**kwargs)

    def propagate(self, model, model_execution):
        """Propagates the walker in phase space using molecular dynamics"""
        device = model_execution.device
        ncores = model_execution.ncores
        dtype  = model_execution.dtype
        def simulate_barebones(walker, model):
            if device == 'cpu':
                torch.set_num_threads(ncores)
            pars = walker.parameters
            np.random.seed(pars.seed)
            torch.manual_seed(pars.seed)
            atoms = walker.state.atoms.copy()
            atoms.calc = model.get_calculator(device, dtype)
            forcefield = create_forcefield(atoms, pars.force_threshold)

            loghook  = yaff.VerletScreenLog(step=pars.step, start=0)
            datahook = DataHook(start=pars.start, step=pars.step)
            hooks = []
            hooks.append(loghook)
            hooks.append(datahook)
            if pars.plumed_input is not None: # add bias if present
                try_manual_plumed_linking()
                with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
                    f.write(pars.plumed_input)
                part_plumed = yaff.external.ForcePartPlumed(
                        forcefield.system,
                        timestep=pars.timestep * molmod.units.femtosecond,
                        restart=0,
                        fn=f.name,
                        fn_log='plumed.log',
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

            # integration
            sample = deepcopy(walker.state) # reset to last saved state
            verlet = yaff.VerletIntegrator(
                    forcefield,
                    timestep=pars.timestep*molmod.units.femtosecond,
                    hooks=hooks,
                    temp0=pars.initial_temperature,
                    )
            if pars.plumed_input is not None:
                os.unlink(f.name) # plumed input file has to exist until here
            yaff.log.set_level(yaff.log.medium)
            try:
                verlet.run(pars.steps)
            except ForceThresholdExceededException as e:
                print(e)
                print('tagging sample as unsafe')
                sample.tag('unsafe')
            yaff.log.set_level(yaff.log.silent)

            # update walker state with last stored state 
            sample.atoms.set_positions(datahook.data[-1].get_positions())
            sample.atoms.set_cell(datahook.data[-1].get_cell())
            walker.state = sample
            #os.unlink('plumed.log') # remove default log file
            return walker
        simulate_electron = ct.electron(
                simulate_barebones,
                executor=model_execution.executor,
                )
        return simulate_electron(self, model)

    @ct.electron(executor='local')
    def reset(self):
        self.state = deepcopy(self.start)
        return self

    @ct.electron(executor='local')
    def sample(self):
        return self.state
