import requests
import logging
from pathlib import Path
import numpy as np

import psiflow
from psiflow.models import NequIPModel, NequIPConfig, MACEModel, MACEConfig
from psiflow.data import Dataset
from psiflow.reference import CP2KReference
from psiflow.sampling import DynamicWalker


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
    train = Dataset.load('data/Al_mil53_train.xyz')
    valid = Dataset.load('data/Al_mil53_valid.xyz')
    model = get_mace_model()
    model.initialize(train)
    model.train(train, valid)
    model.deploy()

    walker = DynamicWalker(train[0], steps=300, step=50)
    _, trajectory = walker.propagate(model=model, keep_trajectory=True)
    reference = get_reference()
    errors = Dataset.get_errors( # compare model and DFT predictions
            reference.evaluate(trajectory),
            model.evaluate(trajectory),
            )
    errors = np.mean(errors.result(), axis=0)
    print('energy error [RMSE, meV/atom]: {}'.format(errors[0]))
    print('forces error [RMSE, meV/A]   : {}'.format(errors[1]))
    print('stress error [RMSE, MPa]     : {}'.format(errors[2]))
    model.save(path_output)


if __name__ == '__main__':
    psiflow.load(
            'local_wq.py',      # path to psiflow config file
            'psiflow_internal', # internal psiflow cache dir
            logging.DEBUG,      # psiflow log level
            logging.INFO,       # parsl log level
            )

    path_output = Path.cwd() / 'output' # stores final model
    main(path_output)
