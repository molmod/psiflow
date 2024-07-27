from ase.units import kJ, mol
import numpy as np

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry
from psiflow.hamiltonians import PlumedHamiltonian, MACEHamiltonian
from psiflow.sampling import Walker, sample, quench, Metadynamics, replica_exchange


PLUMED_INPUT = """UNITS LENGTH=A ENERGY=kj/mol
d_C: DISTANCE ATOMS=3,5
d_O: DISTANCE ATOMS=1,5
CV: COMBINE ARG=d_C,d_O COEFFICIENTS=1,-1 PERIODIC=NO

"""


def get_bias(kappa: float, center: float):
    plumed_str = PLUMED_INPUT
    plumed_str += '\n'
    plumed_str += 'RESTRAINT ARG=CV KAPPA={} AT={}\n'.format(kappa, center)
    return PlumedHamiltonian(plumed_str)


def main():
    aldehyd = Geometry.load('data/acetaldehyde.xyz')
    alcohol = Geometry.load('data/vinyl_alcohol.xyz')

    mace = MACEHamiltonian.mace_cc()
    energy = mace.compute([aldehyd, alcohol], 'energy').result()
    energy = (energy - np.min(energy)) / (kJ / mol)
    print('E_vinyl - E_aldehyde = {:7.3f} kJ/mol'.format(energy[1] - energy[0]))

    # generate initial structures using metadynamics
    plumed_str = PLUMED_INPUT
    plumed_str += 'METAD ARG=CV PACE=10 SIGMA=0.1 HEIGHT=5\n'
    metadynamics = Metadynamics(plumed_str)

    # create 40 identical walkers
    walker = Walker(
        alcohol,
        hamiltonian=mace,
        temperature=300,
        metadynamics=metadynamics,
    )

    # do MTD and create large dataset from all trajectories
    outputs = sample([walker], steps=8000, step=50)
    data_mtd = sum([o.trajectory for o in outputs], start=Dataset([]))
    data_mtd.save('mtd.xyz')

    # initialize walkers for umbrella sampling
    walkers = []
    for i, center in enumerate(np.linspace(1, 3, num=16)):
        bias = get_bias(kappa=1500, center=center)
        hamiltonian = mace + bias
        walker = Walker(alcohol, hamiltonian=hamiltonian, temperature=300)
        walkers.append(walker)
    quench(walkers, data_mtd)  # make sure initial structure is reasonable
    replica_exchange(walkers, trial_frequency=100)  # use REX for improved sampling

    outputs = sample(walkers, steps=1000, step=10)


if __name__ == '__main__':
    with psiflow.load() as f:
        main()
