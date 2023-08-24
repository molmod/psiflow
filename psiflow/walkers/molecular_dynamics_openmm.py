from sys import stdout
from pathlib import Path
import numpy as np
import argparse
import shutil
import torch

import mdtraj
import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from openmmplumed import PlumedForce

from ase import Atoms
from ase.io import read, write
from ase.data import chemical_symbols
from ase.units import nm, fs

from psiflow.walkers.bias import try_manual_plumed_linking
from psiflow.walkers.utils import max_temperature, \
        get_velocities_at_temperature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--ncores', default=None, type=int)
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
    parser.add_argument('--temperature_reset_quantile', default=None, type=float)

    parser.add_argument('--model-cls', default=None, type=str) # model name
    parser.add_argument('--model', default=None, type=str) # model name
    parser.add_argument('--keep-trajectory', default=None, type=bool)
    parser.add_argument('--trajectory', default=None, type=str)
    parser.add_argument('--walltime', default=None, type=float)

    args = parser.parse_args()

    path_plumed = 'plumed.dat'

    assert args.device in ['cpu', 'cuda']
    assert Path(args.atoms).is_file()
    assert args.model_cls in ['MACEModel', 'NequIPModel', 'AllegroModel']
    assert Path(args.model).is_file()
    assert args.trajectory is not None

    import signal
    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    signal.signal(signal.SIGTERM, timeout_handler)

    print('torch: initial num threads: ', torch.get_num_threads())
    torch.set_num_threads(args.ncores)
    print('torch: num threads set to ', torch.get_num_threads())
    if args.force_threshold is not None:
        print('IGNORING requested force threshold at {} eV/A!'.format(args.force_threshold))
        print('force thresholds are only supported in the yaff engine')
    print('device: {}'.format(args.device))
    torch.set_default_dtype(torch.float32)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    atoms = read(args.atoms)
    initial = atoms.copy()

    topology = openmm.app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue('bla', chain)
    for i in range(len(atoms)):
        symbol = chemical_symbols[atoms.numbers[i]]
        element = app.Element.getBySymbol(symbol)
        topology.addAtom(symbol, element, residue)
    if atoms.pbc.all():
        cell = np.array(atoms.cell[:]) * 0.1 # A -> nm
        topology.setPeriodicBoxVectors(cell)
        print('initial periodic box [A]:')
        print(atoms.cell[0, :])
        print(atoms.cell[1, :])
        print(atoms.cell[2, :])
    else:
        print('no periodic boundary conditions applied')
    positions = atoms.positions * 0.1 # A->nm


    A_to_nm          = 0.1
    eV_to_kJ_per_mol = 96.49
    if args.model_cls == 'MACEModel':
        model_cls = 'mace'
        potential = MLPotential(
                model_cls,
                model_path=args.model, 
                distance_to_nm=A_to_nm, 
                energy_to_kJ_per_mol=eV_to_kJ_per_mol,
                )
        system = potential.createSystem(topology, dtype='float32', device=args.device)
    elif args.model_cls in ['NequIPModel', 'AllegroModel']:
        model_cls = 'nequip'
        potential = MLPotential(
                model_cls,
                model_path=args.model, 
                distance_to_nm=A_to_nm, 
                energy_to_kJ_per_mol=eV_to_kJ_per_mol,
                )
        system = potential.createSystem(topology, device=args.device)

    if args.temperature is not None:
        temperature = args.temperature * unit.kelvin
        integrator = openmm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, args.timestep * unit.femtosecond)
        velocities = get_velocities_at_temperature(args.temperature, atoms.get_masses())
    else:
        integrator = openmm.VerletIntegrator(args.timestep * unit.femtosecond)
        velocities = get_velocities_at_temperature(300, atoms.get_masses()) # NVE at 300 K 
    if args.device == 'cuda':
        platform_name = 'CUDA'
    else:
        platform_name = 'CPU'
    platform = openmm.Platform.getPlatformByName(platform_name)
    print('using platform: {}'.format(platform.getName()))
    simulation = app.Simulation(topology, system, integrator, platform=platform)
    simulation.context.setPositions(positions)
    # set velocities in nm / ps
    simulation.context.setVelocities(velocities / nm * (1000 * fs))

    if Path('plumed.dat').is_file():
        try_manual_plumed_linking()
        with open('plumed.dat', 'r') as f:
            plumed_input = f.read()
        system.addForce(PlumedForce(plumed_input))

    if args.pressure is None:
        print('sampling at constant volume ...')
    else:
        print('sampling at constant pressure ...')
        assert args.temperature is not None
        barostat = openmm.MonteCarloFlexibleBarostat(
                10 * args.pressure,          # to bar
                args.temperature,
                False,                       # setScaleMoleculesAsRigid; cannot be kwarg
                )
        system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)


    #simulation.reporters.append(app.PDBReporter('output.pdb', 50, enforcePeriodicBox=True))
    hdf = mdtraj.reporters.HDF5Reporter(
            'traj.h5',
            reportInterval=args.step,
            coordinates=True,
            time=False,
            cell=True,
            potentialEnergy=False,
            temperature=False,
            velocities=True,
            )
    log = app.StateDataReporter(
            stdout,
            args.step,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            elapsedTime=True,
            )
    simulation.reporters.append(hdf)
    simulation.reporters.append(log)

    try:
        simulation.step(args.steps)
        print('completed all steps')
    except TimeoutException:
        print('simulation stopped due to timeout')
    except ValueError as e:
        print(e)
        print('simulation unsafe')
    print('current step: {}'.format(simulation.currentStep))
    hdf.close()

    # save either entire trajectory or only last state
    traj = mdtraj.load_hdf5('traj.h5')
    symbols = list(traj.top.to_dataframe()[0]['element'])
    trajectory = [initial]
    for i in range(traj.xyz.shape[0]):
        if not args.keep_trajectory and (i != traj.xyz.shape[0] - 1):
            continue # only do last frame
        else:
            pass # do all frames
        _atoms = atoms.copy()
        _atoms.set_positions(traj.xyz[i] * 10)
        if atoms.pbc.all():
            _atoms.set_cell(traj.unitcell_vectors[i] * 10)
        trajectory.append(_atoms)
    write(args.trajectory, trajectory)

    # check final temperature
    ekin = simulation.context.getState(getEnergy=True).getKineticEnergy()
    ekin = ekin.value_in_unit(unit.kilojoules_per_mole)
    N  = 3 * len(atoms)
    kB = 8.314 * 1e-3 # in kJ/mol / K
    T  = 2 * ekin / (N * kB)
    print('kinetic energy: ', ekin)
    print('temperature: ', T)

    if (args.temperature_reset_quantile > 0) and (args.temperature is not None):
        T_max = max_temperature(
                args.temperature,
                len(atoms),
                args.temperature_reset_quantile,
                )
        print('T_max: {} K'.format(T_max))
        if T < T_max:
            print('temperature within range')
        else:
            print('temperature outside reasonable range; simulation unsafe')
    else:
        print('no temperature checks performed; simulation assumed to be safe')
