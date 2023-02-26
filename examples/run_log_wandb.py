import requests
import logging
from pathlib import Path
import numpy as np

import psiflow
from psiflow.models import NequIPModel, NequIPConfig, MACEModel, MACEConfig
from psiflow.data import Dataset
from psiflow.reference import CP2KReference
from psiflow.sampling import DynamicWalker, PlumedBias
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


def get_mace_model():
    config = MACEConfig()
    config.max_num_epochs = 1000
    return MACEModel(config)


def main(path_output):
    train = Dataset.load('data/Al_mil53_train.xyz')
    valid = Dataset.load('data/Al_mil53_valid.xyz')
    bias  = get_bias()
    model = get_mace_model()
    model.initialize(train)
    model.deploy()

    wandb_logger = WandBLogger(
            wandb_project='psiflow',
            wandb_group='run_log_wandb',
            error_x_axis='CV',
            )
    log = wandb_logger('untrained', model, data_valid=valid, bias=bias)
    log.result()


if __name__ == '__main__':
    psiflow.load(
            '../configs/local_wq.py',   # path to psiflow config file
            'psiflow_internal',         # internal psiflow cache dir
            logging.DEBUG,              # psiflow log level
            logging.INFO,               # parsl log level
            )

    path_output = Path.cwd() / 'output' # stores final model
    main(path_output)
