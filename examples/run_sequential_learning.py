import requests
import logging
from pathlib import Path
import numpy as np

from ase.io import read

import psiflow
from psiflow.learning import SequentialLearning, load_learning
from psiflow.models import NequIPModel, NequIPConfig
from psiflow.reference import CP2KReference
from psiflow.data import FlowAtoms, Dataset
from psiflow.sampling import BiasedDynamicWalker, PlumedBias
from psiflow.state import load_state        # necessary for restarting a run
from psiflow.wandb_utils import WandBLogger # takes care of W&B logging


def get_bias():
    """Defines the metadynamics parameters based on a plumed input script"""
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=200 HEIGHT=5 PACE=100 LABEL=metad FILE=test_hills
"""
    return PlumedBias(plumed_input)


def get_reference():
    """Defines a generic PBE-D3/TZVP reference level of theory

    Basis set, pseudopotentials, and D3 correction parameters are obtained from
    the official CP2K repository, v9.1, and saved in the internal directory of
    psiflow. The input file is assumed to be available locally.

    """
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


def get_nequip_model():
    """Defines a MACE model architecture

    A full list of parameters can be found in the MACE repository, or in the
    psiflow source code at psiflow.models._nequip.

    """
    config = NequIPConfig()
    config.loss_coeffs['total_energy'][0] = 10
    return NequIPModel(config)


def main(path_output):
    assert not path_output.is_dir()
    reference = get_reference()     # CP2K; PBE-D3(BJ); TZVP
    model = get_nequip_model()      # NequIP; default model
    bias  = get_bias()              # simple MTD bias on unit cell volume
    atoms = read(Path.cwd() / 'data' / 'Al_mil53_train.xyz') # single structure

    # set up wandb logging
    wandb_logger = WandBLogger(
            wandb_project='psiflow',
            wandb_group='run_sequential',
            error_x_axis='CV',  # plot errors against PLUMED variable; 'ARG=CV'
            )

    # set learning parameters and do pretraining
    learning = SequentialLearning(
            path_output=path_output,
            niterations=10,
            train_valid_split=0.9,
            train_from_scratch=True,
            pretraining_nstates=50,
            pretraining_amplitude_pos=0.1,
            pretraining_amplitude_box=0.05,
            use_formation_energy=True,
            wandb_logger=wandb_logger,
            )

    # construct walkers; biased MTD MD in this case
    walkers = BiasedDynamicWalker.multiply(
            30,
            data_start=Dataset([atoms]),
            bias=bias,
            timestep=0.5,
            steps=400,
            step=50,
            start=0,
            temperature=600,
            pressure=0, # NPT
            force_threshold=30,
            initial_temperature=600,
            )
    data_train, data_valid = learning.run(
            model=model,
            reference=reference,
            walkers=walkers,
            )


def restart(path_output):
    reference = get_reference()
    learning  = load_learning(path_output)
    model, walkers, data_train, data_valid, checks = load_state(path_output, '5')
    learning.checks = checks
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
