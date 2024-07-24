from pathlib import Path

import psiflow
from psiflow.reference import CP2K
from psiflow.data import Dataset
from psiflow.sampling import Walker, randomize
from psiflow.models import MACE
from psiflow.hamiltonians import MACEHamiltonian
from psiflow.learning import Learning


def main():
    path_output = Path.cwd() / 'output'

    with open('data/cp2k_input.txt', 'r') as f: cp2k_input = f.read()
    cp2k = CP2K(cp2k_input)

    model = MACE(
        batch_size=4,
        lr=0.02,
        max_ell=3,
        r_max=6.5,
        energy_weight=100,
        correlation=3,
        max_L=1,
        num_channels=24,
        patience=8,
        scheduler_patience=4,
    )
    model.add_atomic_energy('H',  cp2k.compute_atomic_energy('H', box_size=9))
    model.add_atomic_energy('O',  cp2k.compute_atomic_energy('O', box_size=9))

    data = Dataset.load('data/water_train.xyz').filter('energy')
    walkers = (
        Walker(data[0], temperature=300, pressure=0.1).multiply(40) +
        Walker(data[0], temperature=450, pressure=0.1).multiply(40) +
        Walker(data[0], temperature=600, pressure=0.1).multiply(40)
    )
    randomize(walkers, data)  # random initialization
    learning = Learning(
        cp2k,
        path_output,
        wandb_project='psiflow_examples',
        wandb_group='my_water_test',
        initial_data=data,
    )

    model, walkers = learning.passive_learning(
            model,
            walkers,
            hamiltonian=MACEHamiltonian.mace_mp0(),
            steps=10000,
            step=2000,
            )

    for i in range(3):
        model, walkers = learning.active_learning(
                model,
                walkers,
                steps=500,
                )

    # PIMD phase for low-temperature walkers
    for j, walker in walkers[:20]:
        walker.nbeads = 8
    model, walkers = learning.active_learning(
        model,
        walkers,
        steps=1000,
    )


if __name__ == '__main__':
    with psiflow.load():
        main()
