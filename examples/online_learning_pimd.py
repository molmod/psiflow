from pathlib import Path

import psiflow
from psiflow.reference import CP2K
from psiflow.data import Dataset
from psiflow.sampling import Walker
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
        max_num_epochs=200,
    )
    model.add_atomic_energy('H', cp2k.compute_atomic_energy('H', box_size=9))
    model.add_atomic_energy('O', cp2k.compute_atomic_energy('O', box_size=9))

    state = Dataset.load('data/water_train.xyz')[0]
    walkers = (
        Walker(state, temperature=300, pressure=0.1).multiply(40) +
        Walker(state, temperature=450, pressure=0.1).multiply(40) +
        Walker(state, temperature=600, pressure=0.1).multiply(40)
    )
    learning = Learning(
        cp2k,
        path_output,
        wandb_project='psiflow_examples',
        wandb_group='water_learning_pimd',
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
            steps=2000,
        )

    # PIMD phase for low-temperature walkers
    for j, walker in enumerate(walkers[:40]):
        walker.nbeads = 32
    model, walkers = learning.active_learning(
        model,
        walkers,
        steps=500,
    )


if __name__ == '__main__':
    with psiflow.load():
        main()
