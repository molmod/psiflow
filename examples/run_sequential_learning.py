import requests
import logging
from pathlib import Path
import numpy as np

from ase.io import read

import psiflow
from psiflow.learning import SequentialLearning, load_learning
from psiflow.models import NequIPModel, NequIPConfig, MACEModel, MACEConfig
from psiflow.reference import CP2KReference
from psiflow.data import FlowAtoms, Dataset
from psiflow.sampling import DynamicWalker, PlumedBias
from psiflow.generator import Generator
from psiflow.state import load_state
from psiflow.wandb_utils import WandBLogger


def get_bias():
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=200 HEIGHT=5 PACE=100 LABEL=metad FILE=test_hills
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


def get_nequip_model():
    config = NequIPConfig()
    config.loss_coeffs['total_energy'][0] = 10
    return NequIPModel(config)


def get_mace_model():
    config = MACEConfig()
    config.max_num_epochs = 1000
    return MACEModel(config)


def main(path_output):
    assert not path_output.is_dir()
    reference = get_reference() # CP2K; PBE-D3(BJ); TZVP
    model = get_mace_model()    # MACE; small model
    bias  = get_bias()          # simple MTD bias on unit cell volume
    atoms = read(Path.cwd() / 'data' / 'Al_mil53_train.xyz') # load single atoms

    # set up wandb logging
    wandb_logger = WandBLogger(
            wandb_project='psiflow',
            wandb_group='run_sequential',
            error_x_axis='CV',  # plot errors against PLUMED 'ARG=CV'
            )

    # set learning parameters
    learning = SequentialLearning(
            path_output=path_output,
            niterations=10,
            retrain_model_per_iteration=True,
            pretraining_amplitude_pos=0.1,
            pretraining_amplitude_box=0.05,
            pretraining_nstates=50,
            train_valid_split=0.9,
            use_formation_energy=True,
            wandb_logger=wandb_logger,
            )
    data_train, data_valid = learning.run_pretraining(
            model=model,
            reference=reference,
            initial_data=Dataset([atoms]), # only one initial state
            )

    # construct generators; biased MD in this case
    walker = DynamicWalker(
            atoms,
            timestep=0.5,
            steps=400,
            step=50,
            start=0,
            temperature=600,
            pressure=0, # NPT
            force_threshold=30,
            initial_temperature=600,
            )
    generators = Generator('mtd', walker, bias).multiply(30, initialize_using=None)
    data_train, data_valid = learning.run(
            model=model,
            reference=reference,
            generators=generators,
            data_train=data_train,
            data_valid=data_valid,
            )


def restart(path_output):
    reference = get_reference()
    learning  = load_learning(path_output)
    model, generators, data_train, data_valid, checks = load_state(path_output, '5')
    data_train, data_valid = learning.run(
            model=model,
            reference=reference,
            generators=generators,
            data_train=data_train,
            data_valid=data_valid,
            checks=checks,
            )


if __name__ == '__main__':
    psiflow.load(
            'local_htex.py',    # path to psiflow config file
            'psiflow_internal', # internal psiflow cache dir
            logging.DEBUG,      # psiflow log level
            logging.INFO,       # parsl log level
            )

    path_output = Path.cwd() / 'output' # stores learning results
    main(path_output)
    #restart(path_output)
