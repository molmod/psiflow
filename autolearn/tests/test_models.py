import pytest
import yaml
import requests
import numpy as np
import torch

import covalent as ct
c = ct.get_config() # avoid creating a results dir each time tests are executed
c['dispatcher']['results_dir'] = './'
#ct.set_config(c)

from ase.data import chemical_symbols

from autolearn import Dataset, TrainingExecution, ModelExecution
from autolearn.models import NequIPModel
import autolearn.models._nequip

from utils import generate_dummy_data


def test_nequip_config(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['root'] = str(tmp_path)
    model = NequIPModel(config)
    #print(ct.get_config())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires GPU')
def test_nequip_train(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['root'] = str(tmp_path)

    n_train = 5
    n_val   = 1
    natoms  = 5
    atomic_number = 1
    config['chemical_symbols'] = chemical_symbols[atomic_number]
    config['n_train'] = n_train
    config['n_val'] = n_val
    config['max_epochs'] = 10
    model = NequIPModel(config)

    # generate dummy data
    training   = Dataset.from_atoms_list(
            generate_dummy_data(natoms, n_train, atomic_number),
            )
    validation = Dataset.from_atoms_list(
            generate_dummy_data(natoms, n_val,   atomic_number),
            )

    # initialize and train
    model.initialize(training)
    training_execution = TrainingExecution(device='cuda')
    model = model.train(training, validation, training_execution)
    model = model.train(training, validation, training_execution) # continue


def test_nequip_calculator(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['root'] = str(tmp_path)

    n_train = 5
    n_val   = 1
    natoms  = 5
    nstates = n_train + n_val
    atomic_number = 1
    config['chemical_symbols'] = chemical_symbols[atomic_number]
    config['n_train'] = n_train
    config['n_val'] = n_val
    config['max_epochs'] = 10
    model = NequIPModel(config)

    # generate dummy data and initialize model
    training   = Dataset.from_atoms_list(
            generate_dummy_data(natoms, n_train, atomic_number),
            )
    validation = Dataset.from_atoms_list(
            generate_dummy_data(natoms, n_val,   atomic_number),
            )
    model.initialize(training)

    # get calculator
    atoms = training.as_atoms_list()[0].copy()
    atoms.calc = model.get_calculator(
            'cpu',
            'float32',
            )
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    torch.set_default_dtype(torch.double)
    atoms.calc = model.get_calculator(
            'cuda',
            'float64',
            )
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    np.testing.assert_almost_equal(e0, e1, decimal=4)
    np.testing.assert_almost_equal(f0, f1, decimal=4)


def test_model_evaluate(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['root'] = str(tmp_path)

    n_train = 2
    natoms  = 5
    atomic_number = 1
    training   = Dataset.from_atoms_list(
            generate_dummy_data(natoms, n_train, atomic_number),
            )
    model = NequIPModel(config)
    model.initialize(training)

    # generate test data
    n_test = 5
    test = Dataset.from_atoms_list(
            generate_dummy_data(natoms, n_test, atomic_number),
            )
    model_execution = ModelExecution()
    test_evaluated = model.evaluate(test, model_execution)

    # double check using model calculator
    index = -1
    e0    = test_evaluated.as_atoms_list()[index].info['energy']
    atoms = training.as_atoms_list()[0]
    atoms.calc = model.get_calculator(
            model_execution.device,
            model_execution.dtype,
            )
    atoms.set_positions(test_evaluated.as_atoms_list()[index].get_positions())
    atoms.set_cell(test_evaluated.as_atoms_list()[index].get_cell())
    assert np.allclose(atoms.get_potential_energy(), e0)
