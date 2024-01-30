import argparse
import os
from pathlib import Path

import molmod
import numpy as np
import torch
import yaff
from ase import Atoms
from ase.geometry import Cell
from ase.geometry.geometry import find_mic
from ase.io import read, write
from ase.units import Bohr, Ha

from psiflow.walkers.bias import try_manual_plumed_linking


class ForcePartPlumed(yaff.external.ForcePartPlumed):
    """Remove timer from _internal_compute to avoid pathological errors"""

    def _internal_compute(self, gpos, vtens):
        self.plumed.cmd("setStep", self.plumedstep)
        self.plumed.cmd("setPositions", self.system.pos)
        self.plumed.cmd("setMasses", self.system.masses)
        if self.system.charges is not None:
            self.plumed.cmd("setCharges", self.system.charges)
        if self.system.cell.nvec > 0:
            rvecs = self.system.cell.rvecs.copy()
            self.plumed.cmd("setBox", rvecs)
        # PLUMED always needs arrays to write forces and virial to, so
        # provide dummy arrays if Yaff does not provide them
        # Note that gpos and forces differ by a minus sign, which has to be
        # corrected for when interacting with PLUMED
        if gpos is None:
            my_gpos = np.zeros(self.system.pos.shape)
        else:
            gpos[:] *= -1.0
            my_gpos = gpos
        self.plumed.cmd("setForces", my_gpos)
        if vtens is None:
            my_vtens = np.zeros((3, 3))
        else:
            my_vtens = vtens
        self.plumed.cmd("setVirial", my_vtens)
        # Do the actual calculation, without an update; this should
        # only be done at the end of a time step
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalcNoUpdate")
        if gpos is not None:
            gpos[:] *= -1.0
        # Retrieve biasing energy
        energy = np.zeros((1,))
        self.plumed.cmd("getBias", energy)
        return energy[0]


class ForceThresholdExceededException(Exception):
    pass


class ForcePartASE(yaff.pes.ForcePart):
    """YAFF Wrapper around an ASE calculator"""

    def __init__(self, system, atoms):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        atoms : ase.Atoms
            atoms object with calculator included.

        force_threshold : float [eV/A]

        """
        yaff.pes.ForcePart.__init__(self, "ase", system)
        self.system = system  # store system to obtain current pos and box
        self.atoms = atoms

    def _internal_compute(self, gpos=None, vtens=None):
        self.atoms.set_positions(self.system.pos * Bohr)
        if self.atoms.pbc.all():
            self.atoms.set_cell(Cell(self.system.cell._get_rvecs() * Bohr))
        energy = self.atoms.get_potential_energy() / Ha
        if gpos is not None:
            forces = self.atoms.get_forces()
            gpos[:] = -forces / (Ha / Bohr)
        if vtens is not None:
            if self.atoms.pbc.all():
                stress = self.atoms.get_stress(voigt=False)
                volume = np.linalg.det(self.atoms.get_cell())
                vtens[:] = volume * stress / Ha
            else:
                vtens[:] = 0.0
        return energy


class ForceField(yaff.pes.ForceField):
    """Implements force threshold check"""

    def __init__(self, *args, force_threshold=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_threshold = force_threshold

    def _internal_compute(self, gpos, vtens):
        if self.needs_nlist_update:  # never necessary?
            self.nlist.update()
            self.needs_nlist_update = False
        result = sum([part.compute(gpos, vtens) for part in self.parts])
        forces = (-1.0) * gpos * (Ha / Bohr)
        self.check_threshold(forces)
        return result

    def check_threshold(self, forces):
        max_force = np.max(np.linalg.norm(forces, axis=1))
        index = np.argmax(np.linalg.norm(forces, axis=1))
        if max_force > self.force_threshold:
            raise ForceThresholdExceededException(
                "Max force exceeded: {} eV/A by atom index {}".format(max_force, index),
            )


def create_forcefield(atoms, force_threshold):
    """Creates force field from ASE atoms instance"""
    if atoms.pbc.all():
        rvecs = atoms.get_cell() / Bohr
    else:
        rvecs = None
    system = yaff.System(
        numbers=atoms.get_atomic_numbers(),
        pos=atoms.get_positions() / Bohr,
        rvecs=rvecs,
    )
    system.set_standard_masses()
    part_ase = ForcePartASE(system, atoms)
    return ForceField(system, [part_ase], force_threshold=force_threshold)


class DataHook(yaff.VerletHook):
    def __init__(self, start=0, step=1):
        super().__init__(start, step)
        self.atoms = None
        self.data = []

    def init(self, iterative):
        if iterative.ff.system.cell.nvec > 0:
            cell = iterative.ff.system.cell._get_rvecs() * Bohr
        else:
            cell = None
        self.atoms = Atoms(
            numbers=iterative.ff.system.numbers.copy(),
            positions=iterative.ff.system.pos * Bohr,
            cell=cell,
            pbc=cell is not None,
        )

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def __call__(self, iterative):
        self.atoms.set_positions(iterative.ff.system.pos * Bohr)
        if self.atoms.pbc.all():
            self.atoms.set_cell(iterative.ff.system.cell._get_rvecs() * Bohr)
        self.data.append(self.atoms.copy())


class ExtXYZHook(yaff.VerletHook):  # xyz file writer; obsolete
    def __init__(self, path_xyz, start=0, step=1):
        super().__init__(start, step)
        self.path_xyz = path_xyz
        self.atoms = None
        self.nwrites = 0
        self.temperatures = []

    def init(self, iterative):
        if iterative.ff.system.cell.nvec > 0:
            cell = iterative.ff.system.cell._get_rvecs() * Bohr
        else:
            cell = None
        self.atoms = Atoms(
            numbers=iterative.ff.system.numbers.copy(),
            positions=iterative.ff.system.pos * Bohr,
            cell=cell,
            pbc=cell is not None,
        )

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def __call__(self, iterative):
        if iterative.counter > 0:  # first write is manual
            self.atoms.set_positions(iterative.ff.system.pos * Bohr)
            if self.atoms.pbc.all():
                self.atoms.set_cell(iterative.ff.system.cell._get_rvecs() * Bohr)
            write(self.path_xyz, self.atoms, append=True)
            self.nwrites += 1
            self.temperatures.append(iterative.temp)


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
                atoms.get_stress()
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
