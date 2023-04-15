import requests
import logging
from pathlib import Path
import numpy as np

from ase.io import read

import psiflow
from psiflow.learning import ConcurrentLearning, load_learning
from psiflow.models import NequIPModel, NequIPConfig, MACEModel, MACEConfig
from psiflow.reference import CP2KReference
from psiflow.data import FlowAtoms, Dataset
from psiflow.sampling import BiasedDynamicWalker, PlumedBias
from psiflow.state import load_state
from psiflow.wandb_utils import WandBLogger


def get_bias():
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs

coord1: COORDINATION GROUPA=109 GROUPB=88 R_0=1.4
coord2: COORDINATION GROUPA=109 GROUPB=53 R_0=1.4
CV: MATHEVAL ARG=coord1,coord2 FUNC=x-y PERIODIC=NO
cv2: MATHEVAL ARG=coord1,coord2 FUNC=x+y PERIODIC=NO
lwall: LOWER_WALLS ARG=cv2 AT=0.65 KAPPA=5000.0
RESTRAINT ARG=CV AT=0.0 KAPPA=1500.0
"""
    return PlumedBias(plumed_input)


def get_reference():
    with open(Path.cwd() / 'data' / 'cp2k_input.txt', 'r') as f:
        cp2k_input = f.read()
    reference = CP2KReference(cp2k_input=cp2k_input)
    basis     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/BASIS_MOLOPT_UZH').text
    dftd3     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/dftd3.dat').text
    potential = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/POTENTIAL_UZH').text
    cp2k_data = {
            'basis_set': basis,
            'potential': potential,
            'dftd3': dftd3,
            }
    for key, value in cp2k_data.items():
        with open(psiflow.context().path / key, 'w') as f:
            f.write(value)
        reference.add_file(key, psiflow.context().path / key)
    return reference


def get_mace_model():
    config = MACEConfig()
    config.max_num_epochs = 1000
    config.r_max = 6.0
    return MACEModel(config)


def main(path_output):
    assert not path_output.is_dir()
    reference = get_reference() # CP2K; PBE-D3(BJ); TZVP
    model = get_mace_model()    # MACE; small model
    bias  = get_bias()          # simple MTD bias on unit cell volume
    data  = Dataset.load(Path.cwd() / 'data' / 'zeolite_proton.xyz')

    # set up wandb logging
    wandb_logger = WandBLogger(
            wandb_project='psiflow',
            wandb_group='run_concurrent',
            error_x_axis='CV',  # plot errors against PLUMED 'ARG=CV'
            )

    # set learning parameters
    learning = ConcurrentLearning(
            path_output=path_output,
            niterations=10,
            train_from_scratch=True,
            pretraining_amplitude_pos=0.1,
            pretraining_amplitude_box=0.05,
            pretraining_nstates=50,
            train_valid_split=0.9,
            wandb_logger=wandb_logger,
            min_states_per_iteration=15,
            max_states_per_iteration=60,
            )

    # construct walkers; biased MD across the collective variable range of interest
    walkers = BiasedDynamicWalker.distribute(
            20,
            data,
            bias=bias,
            variable='CV',
            min_value=-0.975,
            max_value=0.975,
            timestep=0.5,
            steps=400,
            step=50,
            start=0,
            temperature=1000,
            pressure=0, # NPT
            force_threshold=30,
            initial_temperature=1000,
            )
    data_train, data_valid = learning.run(
            model=model,
            reference=reference,
            walkers=walkers,
            )


def restart(path_output):
    reference = get_reference()
    learning  = load_learning(path_output)
    model, walkers, data_train, data_valid = load_state(path_output, '5')
    data_train, data_valid = learning.run(
            model=model,
            reference=reference,
            walkers=walkers,
            initial_data=data_train + data_valid,
            )


if __name__ == '__main__':
    psiflow.load()

    path_output = Path.cwd() / 'output' # stores learning results
    main(path_output)
    #restart(path_output)
