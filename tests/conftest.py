import pytest
import logging
import parsl
import requests
import yaml
import torch
import numpy as np
import tempfile
from pathlib import Path
from dataclasses import asdict

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from psiflow.execution import ExecutionContext, ModelEvaluationExecution, \
        ModelTrainingExecution, ReferenceEvaluationExecution, \
        generate_parsl_config
from psiflow.utils import get_psiflow_config_from_file
from psiflow.data import Dataset, FlowAtoms
from psiflow.models import NequIPModel, NequIPConfig, MACEModel, MACEConfig, \
        AllegroModel, AllegroConfig
from psiflow.reference import CP2KReference, EMTReference


def pytest_addoption(parser):
    parser.addoption(
            '--psiflow-config',
            action='store',
            #default='local_threadpool',
            help='test',
            )


@pytest.fixture(scope='session')
def context(request, tmp_path_factory):
    psiflow_config_path = Path(request.config.getoption('--psiflow-config'))
    config, definitions = get_psiflow_config_from_file(
            psiflow_config_path,
            tmp_path_factory.mktemp('parsl_internal'),
            )
    parsl.load(config)
    #parsl.set_stream_logger('parsl', level=logging.DEBUG)
    context = ExecutionContext(
            config,
            definitions,
            path=str(tmp_path_factory.mktemp('context_dir')),
            )
    yield context
    from parsl.dataflow.dflow import DataFlowKernelLoader
    #dfk = DataFlowKernelLoader.dfk()
    #dfk.wait_for_current_tasks()
    parsl.clear()


@pytest.fixture
def nequip_config(tmp_path):
    nequip_config = NequIPConfig()
    nequip_config.root = str(tmp_path)
    nequip_config.wandb_group = 'pytest_group'
    return asdict(nequip_config)


@pytest.fixture
def allegro_config(tmp_path):
    allegro_config = AllegroConfig()
    allegro_config.root = str(tmp_path)
    return asdict(allegro_config)


@pytest.fixture
def mace_config(tmp_path):
    mace_config = MACEConfig()
    return asdict(mace_config)


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
