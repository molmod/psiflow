import pytest
import yaml
import requests
import numpy as np
import torch

import covalent as ct
c = ct.get_config() # avoid creating a results dir each time tests are executed
c['dispatcher']['results_dir'] = './'
#ct.set_config(c)

from ase import Atoms
from ase.geometry import Cell
from ase.data import chemical_symbols

from autolearn import Dataset, TrainingExecution, ModelExecution
from autolearn.models import NequIPModel
import autolearn.models._nequip


def generate_dummy_data(natoms, nstates, number):
    atoms = Atoms(
            numbers=number * np.ones(natoms),
            positions=np.random.uniform(0, 1, size=(natoms, 3)),
            )
    data = []
    for i in range(nstates):
        _atoms = atoms.copy()
        _atoms.info['energy'] = np.random.uniform(0, 1)
        _atoms.arrays['forces'] = np.random.uniform(0, 1, size=(natoms, 3))
        _atoms.set_positions(np.random.uniform(0, 1, size=(natoms, 3)))
        data.append(_atoms)
    return data


def test_nequip_config(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['root'] = str(tmp_path)
    model = NequIPModel(config)
    #print(ct.get_config())


def test_nequip_train(tmp_path):
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

    # generate dummy data
    data = generate_dummy_data(natoms, nstates, atomic_number)
    dataset = Dataset(data[:n_train], data[n_train:])

    # initialize and train
    model.initialize(dataset)
    training_execution = TrainingExecution()
    training_execution.device = 'cuda'
    model = NequIPModel.train(model, training_execution, dataset)

    # continue training
    model = NequIPModel.train(model, training_execution, dataset)


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
    data = generate_dummy_data(natoms, nstates, atomic_number)
    dataset = Dataset(data[:n_train], data[n_train:])
    model.initialize(dataset)

    # get calculator
    atoms = data[0]
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


def test_evaluate_nequip(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['root'] = str(tmp_path)

    n_train = 10
    n_val   = 0
    natoms  = 5
    nstates = n_train + n_val
    atomic_number = 1
    data = generate_dummy_data(natoms, nstates, atomic_number)
    dataset = Dataset(data[:n_train], data[n_train:])

    # initialize two different models
    config['chemical_symbols'] = chemical_symbols[atomic_number]
    config['n_train'] = n_train
    config['n_val'] = n_val
    config['seed'] = 1
    pass
    #model  = NequIPModel(config)
    #model.initialize(dataset)

    #config['seed'] = 2
    #model_ = NequIPModel(config)
    #model_.initialize(dataset)

    ## get calculator
    #atoms = data[0]
    #model_execution = ModelExecution() # on cpu
    #atoms.calc = model.get_calculator(model_execution)
    #atoms.get_potential_energy()
