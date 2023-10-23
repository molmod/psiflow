import argparse
import os
from pathlib import Path

import molmod
import numpy as np
import torch
import yaff
from ase.geometry.geometry import find_mic
from ase.io import read, write

from psiflow.walkers.bias import try_manual_plumed_linking
from psiflow.walkers.utils import (
    DataHook,
    ExtXYZHook,
    ForcePartPlumed,
    ForceThresholdExceededException,
    create_forcefield,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--ncores", default=None, type=int)
    parser.add_argument("--atoms", default=None, type=str)

    # pars
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--timestep", default=None, type=float)
    parser.add_argument("--steps", default=None, type=int)
    parser.add_argument("--step", default=None, type=int)
    parser.add_argument("--start", default=None, type=int)
    parser.add_argument("--temperature", default=None, type=float)
    parser.add_argument("--pressure", default=None, type=float)
    parser.add_argument("--force_threshold", default=None, type=float)
    parser.add_argument("--max_excess_temperature", default=None, type=float)
    parser.add_argument("--distance_threshold", default=None, type=float)

    parser.add_argument("--model-cls", default=None, type=str)  # model name
    parser.add_argument("--model", default=None, type=str)  # model name
    parser.add_argument("--keep-trajectory", default=None, type=bool)
    parser.add_argument("--trajectory", default=None, type=str)
    parser.add_argument("--walltime", default=None, type=float)

    args = parser.parse_args()

    path_plumed = "plumed.dat"

    assert args.device in ["cpu", "cuda"]
    assert Path(args.atoms).is_file()
    assert args.model_cls in ["MACEModel", "NequIPModel", "AllegroModel"]
    assert Path(args.model).is_file()
    assert args.trajectory is not None

    import signal

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(args.walltime))

    print("torch: initial num threads: ", torch.get_num_threads())
    torch.set_num_threads(args.ncores)
    print("torch: num threads set to ", torch.get_num_threads())
    torch.set_default_dtype(torch.float32)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    atoms = read(args.atoms)
    initial = atoms.copy()
    if args.model_cls == "MACEModel":
        from psiflow.models import MACEModel

        load_calculator = MACEModel.load_calculator
    elif args.model_cls == "NequIPModel":
        from psiflow.models import NequIPModel

        load_calculator = NequIPModel.load_calculator
    elif args.model_cls == "AllegroModel":
        from psiflow.models import AllegroModel

        load_calculator = AllegroModel.load_calculator
    else:
        raise ValueError
    atoms.calc = load_calculator(args.model, args.device)
    forcefield = create_forcefield(atoms, args.force_threshold)

    loghook = yaff.VerletScreenLog(step=args.step, start=0)
    datahook = DataHook()  # bug in YAFF: override start/step after init
    datahook.start = args.start
    datahook.step = args.step
    hooks = []
    hooks.append(loghook)
    hooks.append(datahook)
    if args.keep_trajectory:
        xyz = ExtXYZHook(args.trajectory)  # assign start/step manually
        xyz.start = args.start
        xyz.step = args.step
        print("XYZ write start: {}".format(xyz.start))
        print("XYZ write step: {}".format(xyz.step))
        hooks.append(xyz)

    # first write is manual
    write(args.trajectory, read(args.atoms))

    if Path("plumed.dat").is_file():
        try_manual_plumed_linking()
        part_plumed = ForcePartPlumed(
            forcefield.system,
            timestep=args.timestep * molmod.units.femtosecond,
            restart=1,
            fn="plumed.dat",
            fn_log=str(Path.cwd() / "plumed_log"),
        )
        forcefield.add_part(part_plumed)
        hooks.append(part_plumed)  # NECESSARY!!

    if args.temperature is not None:
        thermo = yaff.LangevinThermostat(
            args.temperature,
            timecon=100 * molmod.units.femtosecond,
        )
        if args.pressure is None:
            print("sampling NVT ensemble ...")
            hooks.append(thermo)
        else:
            print("sampling NPT ensemble ...")
            try:  # some models do not have stress support; prevent NPT!
                stress = atoms.get_stress()
            except Exception:
                raise ValueError("NPT requires stress support in model")
            baro = yaff.LangevinBarostat(
                forcefield,
                args.temperature,
                args.pressure * 1e6 * molmod.units.pascal,  # in MPa
                timecon=molmod.units.picosecond,
                anisotropic=True,
                vol_constraint=False,
            )
            tbc = yaff.TBCombination(thermo, baro)
            hooks.append(tbc)
    else:
        print("sampling NVE ensemble")

    assert os.path.getsize(args.trajectory) > 0  # should be nonempty!
    initial_size = os.path.getsize(args.trajectory) > 0
    try:  # exception may already be raised at initialization of verlet
        if args.temperature is not None:
            temp0 = args.temperature
        else:
            temp0 = 300
        verlet = yaff.VerletIntegrator(
            forcefield,
            timestep=args.timestep * molmod.units.femtosecond,
            hooks=hooks,
            temp0=temp0,
        )
        yaff.log.set_level(yaff.log.medium)
        verlet.run(args.steps)
        counter = verlet.counter
    except ForceThresholdExceededException as e:
        print(e)
        print("simulation is unsafe")
        counter = 0
    except TimeoutException as e:
        print(e)
    except Exception as e:
        print(e)
        print("simulation is unsafe")
        counter = 0
    yaff.log.set_level(yaff.log.silent)

    if Path("plumed.dat").is_file():
        os.unlink("plumed.dat")

    # update state with last stored state if data nonempty
    atoms.calc = None
    if len(datahook.data) > 0:
        atoms.set_positions(datahook.data[-1].get_positions())
        if atoms.pbc.all():
            atoms.set_cell(datahook.data[-1].get_cell())
    else:
        datahook.data.append(initial)

    # write data to output xyz
    if counter > 0:
        if args.keep_trajectory:
            pass  # already written by ExtXYZHook
        elif os.path.getsize(args.trajectory) > initial_size:
            write(args.trajectory, atoms)
    else:
        if not args.keep_trajectory:
            os.path.remove(args.trajectory)
            write(args.trajectory, read(args.atoms))

    if counter > 0:
        print(
            "check whether all interatomic distances > {}".format(
                args.distance_threshold
            )
        )
        state = atoms
        nrows = int(len(state) * (len(state) - 1) / 2)
        deltas = np.zeros((nrows, 3))
        count = 0
        for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):
                deltas[count] = state.positions[i] - state.positions[j]
                count += 1
        assert count == nrows
        if state.pbc.all():
            deltas, _ = find_mic(deltas, state.cell)
        distances = np.linalg.norm(deltas, axis=1)
        check = np.all(distances > args.distance_threshold)
        if check:
            print("\tOK")
        else:
            print("\tunsafe! Found d = {} A".format(np.min(distances)))

    # perform temperature check
    T = verlet.temp
    print("temperature: ", T)
    if (args.max_excess_temperature is not None) and (args.temperature is not None):
        T_max = args.temperature + args.max_excess_temperature
        print("T_max: {} K".format(T_max))
        if T < T_max:
            print("temperature within range")
        else:
            print("temperature outside reasonable range; simulation unsafe")
    else:
        print("no temperature checks performed; simulation assumed to be safe")
    return None
