import requests     # required for downloading cp2k input
import logging
from pathlib import Path
import numpy as np

import psiflow
from psiflow.models import MACEModel, MACEConfig
from psiflow.data import Dataset
from psiflow.reference import CP2KReference
from psiflow.sampling import DynamicWalker


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


def get_mace_model():
    """Defines a MACE model architecture

    A full list of parameters can be found in the MACE repository, or in the
    psiflow source code at psiflow.models._mace.

    """
    config = MACEConfig()           # load default MACE parameters
    config.max_num_epochs = 1000
    config.r_max = 6.0
    return MACEModel(config)


def main(path_output):
    """Main script

    The following operations are performed:
        - initial data available in .xyz is loaded
        - a MACE model is initialized and trained on this data
        - the model is deployed and used in a short molecular dynamics run
        - the reference object is used to evaluate the trajectory ab initio
        - the errors between the MACE prediction and the reference
          energy/forces/stress is computed.

    """
    train = Dataset.load('data/Al_mil53_train.xyz')
    valid = Dataset.load('data/Al_mil53_valid.xyz')
    model = get_mace_model()
    model.initialize(train)
    model.train(train, valid)
    model.deploy()

    walker = DynamicWalker(train[0], steps=300, step=50)
    _, trajectory = walker.propagate(model=model, keep_trajectory=True)
    reference = get_reference()
    errors = Dataset.get_errors(    # compare model and DFT predictions
            reference.evaluate(trajectory),
            model.evaluate(trajectory),
            metric='rmse',          # 'mae' or 'max' are also possible
            )
    errors = np.mean(errors.result(), axis=0)
    print('energy error [RMSE, meV/atom]: {}'.format(errors[0]))
    print('forces error [RMSE, meV/A]   : {}'.format(errors[1]))
    print('stress error [RMSE, MPa]     : {}'.format(errors[2]))
    model.save(path_output)


if __name__ == '__main__':              # entry point
    psiflow.load()
    path_output = Path.cwd() / 'output' # stores final model
    path_output.mkdir()
    main(path_output)
