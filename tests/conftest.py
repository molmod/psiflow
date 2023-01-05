import pytest
import sys
import parsl
import requests
import yaml
import torch
import numpy as np
import tempfile
import importlib
from pathlib import Path

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from flower.execution import ExecutionContext, TrainingExecutionDefinition, \
        ModelExecutionDefinition, ReferenceExecutionDefinition
from flower.data import Dataset, FlowerAtoms


def pytest_addoption(parser):
    parser.addoption(
            '--parsl-config',
            action='store',
            #default='local_threadpool',
            help='test',
            )


@pytest.fixture(scope='session')
def parsl_config(request, tmpdir_factory):
    parsl_config_path = Path(request.config.getoption('--parsl-config'))
    assert parsl_config_path.is_file()
    # see https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    spec = importlib.util.spec_from_file_location('module.name', parsl_config_path)
    parsl_config_module = importlib.util.module_from_spec(spec)
    sys.modules['module.name'] = parsl_config_module
    spec.loader.exec_module(parsl_config_module)
    return parsl_config_module.get_config(tmpdir_factory.mktemp('parsl_config_dir'))


@pytest.fixture(scope='session')
def context(parsl_config, tmpdir_factory):
    parsl.load(parsl_config)
    path = str(tmpdir_factory.mktemp('internal'))
    context = ExecutionContext(parsl_config, path=path)
    context.register(ModelExecutionDefinition())
    context.register(ReferenceExecutionDefinition(ncores=4, time_per_singlepoint=30))
    context.register(TrainingExecutionDefinition())
    yield context
    parsl.clear()


@pytest.fixture
def nequip_config(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['r_max'] = 3.5 # reduce computational cost of data processing
    config['chemical_symbols'] = ['X'] # should get overridden
    return config


def generate_emt_cu_data(nstates, amplitude):
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.calc = EMT()
    pos = atoms.get_positions()
    box = atoms.get_cell()
    atoms_list = []
    for i in range(nstates):
        atoms.set_positions(pos + np.random.uniform(-amplitude, amplitude, size=(len(atoms), 3)))
        atoms.set_cell(box + np.random.uniform(-amplitude, amplitude, size=(3, 3)))
        _atoms = atoms.copy()
        _atoms.calc = None
        _atoms.info['energy']   = atoms.get_potential_energy()
        _atoms.info['stress']   = atoms.get_stress(voigt=False)
        _atoms.arrays['forces'] = atoms.get_forces()
        # make content heterogeneous to test per_element functions
        _atoms.numbers[0] = 1
        _atoms.symbols[0] = 'H'
        atoms_list.append(_atoms)
    return atoms_list


@pytest.fixture
def dataset(context, tmp_path):
    data = generate_emt_cu_data(20, 0.2)
    data_ = [FlowerAtoms.from_atoms(atoms) for atoms in data]
    for atoms in data_:
        atoms.evaluation_flag = 'success'
    return Dataset(context, data_)
