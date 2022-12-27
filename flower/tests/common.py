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

from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import ThreadPoolExecutor, HighThroughputExecutor
from parsl.launchers import SimpleLauncher, SingleNodeLauncher

from flower.execution import ExecutionContext, TrainingExecutionDefinition, \
        ModelExecutionDefinition, ReferenceExecutionDefinition, \
        DefaultExecutionDefinition


@pytest.fixture(scope='module', params=['threadpool'])
def context(request, tmpdir_factory):
    if request.param == 'threadpool':
        executors = [
                ThreadPoolExecutor(label='gpu', max_threads=1, working_dir=str(tmpdir_factory.mktemp('working_dir'))),
                ThreadPoolExecutor(label='default', max_threads=1, working_dir=str(tmpdir_factory)),
                ThreadPoolExecutor(label='cpu_small', max_threads=4, working_dir=str(tmpdir_factory)),
                ThreadPoolExecutor(label='cpu_large', max_threads=4, working_dir=str(tmpdir_factory)),
                ]
    elif request.param == 'htex':
        provider = LocalProvider(
            min_blocks=1,
            max_blocks=1,
            nodes_per_block=1,
            parallelism=0.5,
            launcher=SingleNodeLauncher(),
            )
        executors = [
                HighThroughputExecutor(address='localhost', label='gpu', working_dir=str(tmpdir_factory), provider=provider, max_workers=1),
                HighThroughputExecutor(address='localhost', label='default', working_dir=str(tmpdir_factory), provider=provider, cores_per_worker=1),
                HighThroughputExecutor(address='localhost', label='cpu_small', working_dir=str(tmpdir_factory), provider=provider),
                HighThroughputExecutor(address='localhost', label='cpu_large', working_dir=str(tmpdir_factory), provider=provider, max_workers=1, cores_per_worker=4),
                ]
    else:
        raise ValueError
    config = Config(executors, run_dir=str(tmpdir_factory.mktemp('runinfo')))
    path = tempfile.mkdtemp()
    parsl.load(config)
    context = ExecutionContext(config, path=path)
    context.register(DefaultExecutionDefinition())
    model_execution = ModelExecutionDefinition()
    context.register(model_execution)
    context.register(ReferenceExecutionDefinition(ncores=4))
    if torch.cuda.is_available(): # requires gpu
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


def generate_emt_cu_data(nstates):
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.calc = EMT()
    pos = atoms.get_positions()
    box = atoms.get_cell()
    atoms_list = []
    amplitude = 0.2
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
