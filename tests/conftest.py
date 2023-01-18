import pytest
import parsl
import requests
import yaml
import torch
import numpy as np
import tempfile
from pathlib import Path

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from psiflow.execution import ExecutionContext, TrainingExecutionDefinition, \
        ModelExecutionDefinition, ReferenceExecutionDefinition
from psiflow.data import Dataset, FlowAtoms
from psiflow.utils import get_parsl_config_from_file


def pytest_addoption(parser):
    parser.addoption(
            '--parsl-config',
            action='store',
            #default='local_threadpool',
            help='test',
            )


@pytest.fixture(scope='session')
def parsl_config(request, tmp_path_factory):
    parsl_config_path = Path(request.config.getoption('--parsl-config'))
    return get_parsl_config_from_file(
            parsl_config_path,
            tmp_path_factory.mktemp('parsl_internal'),
            )


@pytest.fixture(scope='session')
def context(parsl_config, tmpdir_factory):
    parsl_config.retries = 0
    parsl.load(parsl_config)
    path = str(tmpdir_factory.mktemp('context_dir'))
    context = ExecutionContext(parsl_config, path=path)
    context.register(ModelExecutionDefinition())
    context.register(ReferenceExecutionDefinition(
        time_per_singlepoint=50,
        mpi_command=lambda x: 'mympirun ',
        ))
    context.register(TrainingExecutionDefinition(walltime=50)) # large for vsc
    yield context
    parsl.clear()


@pytest.fixture
def nequip_config(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['r_max'] = 3.5 # reduce computational cost of data processing
    config['chemical_symbols'] = ['X'] # should get overridden
    config['l_max'] = 1
    config['max_epochs'] = 10000
    config['wandb'] = True
    config['wandb_project'] = 'pytest-nequip'
    config['metrics_components'] = [
            ['forces', 'mae'],
            #['forces', 'rmse'],
            ['forces', 'mae', {'PerSpecies': True, 'report_per_component': False}],
            ['total_energy', 'mae', {'PerAtom': True}],
            ]
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
    data_ = [FlowAtoms.from_atoms(atoms) for atoms in data]
    for atoms in data_:
        atoms.reference_status = True
    return Dataset(context, data_)
