import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import parsl
import pytest
import requests
import yaml
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

import psiflow
from psiflow.data import Dataset, FlowAtoms
from psiflow.models import (
    AllegroConfig,
    AllegroModel,
    MACEConfig,
    MACEModel,
    NequIPConfig,
    NequIPModel,
)
from psiflow.reference import EMTReference


def pytest_addoption(parser):
    parser.addoption(
        "--psiflow-config",
        action="store",
        help="test",
    )
    parser.addoption(
        "--skip-gpu",
        action="store_true",
        default=False,
        help="whether to run tests which require a GPU",
        )


@pytest.fixture(scope="session")
def gpu(request):
    if request.config.getoption("--skip-gpu"):
        pytest.skip('skipping tests which require GPU')


@pytest.fixture(scope="session", autouse=True)
def context(request, tmp_path_factory):
    path_config = Path(request.config.getoption("--psiflow-config"))
    try:
        context = psiflow.context()
    except RuntimeError:
        context = psiflow.load(
            path_config,
            tmp_path_factory.mktemp("psiflow_internal"),
            psiflow_log_level="INFO",
            parsl_log_level="WARN",
        )

    def cleanup():
        parsl.dfk().wait_for_current_tasks()

    request.addfinalizer(cleanup)
    return context


@pytest.fixture(scope='session')
def nequip_config(tmp_path_factory):
    nequip_config = NequIPConfig()
    nequip_config.root = str(tmp_path_factory.mktemp('nequip_config_temp'))
    nequip_config.wandb_group = "pytest_group"
    nequip_config.num_layers = 1
    nequip_config.num_features = 2
    nequip_config.num_basis = 2
    nequip_config.invariant_layers = 1
    nequip_config.invariant_neurons = 2
    nequip_config.r_max = 4
    return asdict(nequip_config)


@pytest.fixture(scope='session')
def allegro_config(tmp_path_factory):
    allegro_config = AllegroConfig()
    allegro_config.root = str(tmp_path_factory.mktemp('allegro_config_temp'))
    allegro_config.env_embed_multiplicity = 2
    allegro_config.two_body_latent_mlp_latent_dimensions = [2, 2, 4]
    allegro_config.mlp_latent_dimensions = [4]
    allegro_config.r_max = 4
    return asdict(allegro_config)


@pytest.fixture(scope='session')
def mace_config():
    mace_config = MACEConfig()
    mace_config.num_radial_basis = 3
    mace_config.num_cutoff_basis = 2
    mace_config.max_ell = 1
    mace_config.correlation = 1
    mace_config.MLP_irreps = "2x0e"
    mace_config.num_channels = 2
    mace_config.max_L = 0
    mace_config.r_max = 4
    mace_config.radial_MLP = "[4]"
    return asdict(mace_config)


def generate_emt_cu_data(nstates, amplitude):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = EMT()
    pos = atoms.get_positions()
    box = atoms.get_cell()
    atoms_list = []
    for i in range(nstates):
        atoms.set_positions(
            pos + np.random.uniform(-amplitude, amplitude, size=(len(atoms), 3))
        )
        atoms.set_cell(box + np.random.uniform(-amplitude, amplitude, size=(3, 3)))
        _atoms = atoms.copy()
        _atoms.calc = None
        _atoms.info["energy"] = atoms.get_potential_energy()
        _atoms.info["stress"] = atoms.get_stress(voigt=False)
        _atoms.arrays["forces"] = atoms.get_forces()
        # make content heterogeneous to test per_element functions
        _atoms.numbers[0] = 1
        _atoms.symbols[0] = "H"
        atoms_list.append(_atoms)
    return atoms_list


@pytest.fixture
def dataset(context):
    data = generate_emt_cu_data(20, 0.2)
    data_ = [FlowAtoms.from_atoms(atoms) for atoms in data]
    for atoms in data_:
        atoms.reference_status = True
    return Dataset(data_).canonical_orientation()


@pytest.fixture(scope='session')
def mace_model(mace_config):
    # manually recreate dataset with 'session' scope
    data = generate_emt_cu_data(20, 0.2)
    data_ = [FlowAtoms.from_atoms(atoms) for atoms in data]
    for atoms in data_:
        atoms.reference_status = True
    dataset = Dataset(data_)
    model = MACEModel(mace_config)
    # add additional state to initialize other atomic numbers
    # mace cannot handle partially periodic datasets
    atoms = Atoms(
            numbers=2 * [101],
            positions=np.array([[0, 0, 0], [2, 0, 0]]),
            cell=2 * np.eye(3),
            pbc=True,
    )
    atoms.info['energy'] = -1.0
    atoms.arrays['forces'] = np.random.uniform(size=(2, 3))
    atoms = FlowAtoms.from_atoms(atoms)
    atoms.reference_status = True
    model.initialize(dataset[:5] + Dataset([atoms]))
    psiflow.wait()
    return model


@pytest.fixture
def dataset_h2(context):
    h2 = FlowAtoms(
        numbers=[1, 1],
        positions=[[0, 0, 0], [0.74, 0, 0]],
        pbc=False,
    )
    data = [h2.copy() for i in range(20)]
    for atoms in data:
        atoms.set_positions(
            atoms.get_positions() + np.random.uniform(-0.05, 0.05, size=(2, 3))
        )
    return Dataset(data)
