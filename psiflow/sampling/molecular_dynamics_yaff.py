import torch
import argparse
import os
import tempfile
import numpy as np
from copy import deepcopy
from pathlib import Path

import yaff
import molmod
from ase.io import read, write
from ase.io.extxyz import write_extxyz

from psiflow.data import FlowAtoms
from psiflow.sampling.utils import ForcePartASE, DataHook, \
        create_forcefield, ForceThresholdExceededException, ForcePartPlumed
from psiflow.sampling.bias import try_manual_plumed_linking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--ncores', default=None, type=int)
    parser.add_argument('--dtype', default=None, type=str)
    parser.add_argument('--atoms', default=None, type=str)

    # pars
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--timestep', default=None, type=float)
    parser.add_argument('--steps', default=None, type=int)
    parser.add_argument('--step', default=None, type=int)
    parser.add_argument('--start', default=None, type=int)
    parser.add_argument('--temperature', default=None, type=float)
    parser.add_argument('--pressure', default=None, type=float)
    parser.add_argument('--force_threshold', default=None, type=float)
    parser.add_argument('--initial_temperature', default=None, type=float)

    parser.add_argument('--model-cls', default=None, type=str) # model name
    parser.add_argument('--model', default=None, type=str) # model name
    parser.add_argument('--keep-trajectory', default=None, type=bool)
    parser.add_argument('--trajectory', default=None, type=str)
    parser.add_argument('--walltime', default=None, type=float)

    args = parser.parse_args()

    path_plumed = 'plumed.dat'

    assert args.device in ['cpu', 'cuda']
    assert args.dtype in ['float32', 'float64']
    assert Path(args.atoms).is_file()
    assert args.model_cls in ['MACEModel', 'NequIPModel', 'AllegroModel']
    assert Path(args.model).is_file()
    assert args.trajectory is not None

    import signal
    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(args.walltime))

    print('torch: initial num threads: ', torch.get_num_threads())
    torch.set_num_threads(args.ncores)
    print('torch: num threads set to ', torch.get_num_threads())
    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    atoms = read(args.atoms)
    initial = atoms.copy()
    if args.model_cls == 'MACEModel':
        from psiflow.models import MACEModel
        load_calculator = MACEModel.load_calculator
    elif args.model_cls == 'NequIPModel':
        from psiflow.models import NequIPModel
        load_calculator = NequIPModel.load_calculator
    elif args.model_cls == 'AllegroModel':
        from psiflow.models import AllegroModel
        load_calculator = AllegroModel.load_calculator
    else:
        raise ValueError
    atoms.calc = load_calculator(args.model, args.device, args.dtype)
    forcefield = create_forcefield(atoms, args.force_threshold)

    loghook  = yaff.VerletScreenLog(step=args.step, start=0)
    datahook = DataHook() # bug in YAFF: override start/step after init
    datahook.start = args.start
    datahook.step  = args.step
    hooks = []
    hooks.append(loghook)
    hooks.append(datahook)
    if Path('plumed.dat').is_file():
        try_manual_plumed_linking()
        part_plumed = ForcePartPlumed(
                forcefield.system,
                timestep=args.timestep * molmod.units.femtosecond,
                restart=1,
                fn='plumed.dat',
                fn_log=str(Path.cwd() / 'plumed_log'),
                )
        forcefield.add_part(part_plumed)
        hooks.append(part_plumed) # NECESSARY!!

    if args.temperature is not None:
        thermo = yaff.LangevinThermostat(
                args.temperature,
                timecon=100 * molmod.units.femtosecond,
                )
        if args.pressure is None:
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
                    args.temperature,
                    args.pressure * 1e6 * molmod.units.pascal, # in MPa
                    timecon=molmod.units.picosecond,
                    anisotropic=True,
                    vol_constraint=False,
                    )
            tbc = yaff.TBCombination(thermo, baro)
            hooks.append(tbc)
    else:
        print('sampling NVE ensemble')

    tag = 'safe'
    counter = 0
    try: # exception may already be raised at initialization of verlet
        verlet = yaff.VerletIntegrator(
                forcefield,
                timestep=args.timestep*molmod.units.femtosecond,
                hooks=hooks,
                temp0=args.initial_temperature,
                )
        yaff.log.set_level(yaff.log.medium)
        verlet.run(args.steps)
        counter = verlet.counter
    except ForceThresholdExceededException as e:
        print(e)
        print('tagging sample as unsafe')
        tag = 'unsafe'
        try:
            counter = verlet.counter
        except UnboundLocalError: # if it happened during verlet init
            pass
    except TimeoutException as e:
        counter = verlet.counter
        print(e)
    yaff.log.set_level(yaff.log.silent)

    if Path('plumed.dat').is_file():
        os.unlink('plumed.dat')

    # update state with last stored state if data nonempty
    atoms.calc = None
    if len(datahook.data) > 0:
        atoms.set_positions(datahook.data[-1].get_positions())
        atoms.set_cell(datahook.data[-1].get_cell())
    else:
        datahook.data.append(initial)

    # write data to output xyz
    if args.keep_trajectory:
        with open(args.trajectory, 'w+') as f:
            write_extxyz(f, datahook.data)
    else:
        write(args.trajectory, datahook.data[-1])
    assert os.path.getsize(args.trajectory) > 0 # should be nonempty!

    # check whether counter == 0 actually means state = start
    counter_is_reset = counter == 0
    state_is_reset   = np.allclose(
                initial.get_positions(),
                atoms.get_positions(),
                )
    if counter_is_reset: assert state_is_reset
    if state_is_reset and (args.step == 1): assert counter_is_reset
    return FlowAtoms.from_atoms(atoms), tag, counter
