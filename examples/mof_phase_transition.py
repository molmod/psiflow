import requests
import logging
from pathlib import Path
import numpy as np

from ase.io import read

import psiflow
from psiflow.learning import SequentialLearning, load_learning
from psiflow.models import MACEModel, MACEConfig
from psiflow.reference import CP2KReference
from psiflow.data import FlowAtoms, Dataset
from psiflow.walkers import BiasedDynamicWalker, PlumedBias
from psiflow.state import load_state
from psiflow.metrics import Metrics


def get_bias():
    """Defines the metadynamics parameters based on a plumed input script"""
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=200 HEIGHT=5 PACE=100 LABEL=metad FILE=test_hills
"""
    return PlumedBias(plumed_input)


def get_reference():
    """Defines a generic PBE-D3/TZVP reference level of theory

    Basis set, pseudopotentials, and D3 correction parameters are obtained from
    the official CP2K repository, v2023.1, and saved in the internal directory of
    psiflow. The input file is assumed to be available locally.

    """
    with open(Path.cwd() / 'data' / 'cp2k_input.txt', 'r') as f:
        cp2k_input = f.read()
    reference = CP2KReference(cp2k_input=cp2k_input)
    basis     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v2023.1/data/BASIS_MOLOPT_UZH').text
    dftd3     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v2023.1/data/dftd3.dat').text
    potential = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v2023.1/data/POTENTIAL_UZH').text
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


def main(path_output):
    assert not path_output.exists()
    reference = get_reference()     # CP2K; PBE-D3(BJ); TZVP
    bias      = get_bias()          # simple MTD bias on unit cell volume
    atoms = FlowAtoms.from_atoms(read(Path.cwd() / 'data' / 'mof.xyz'))
    atoms.canonical_orientation()   # transform into conventional lower-triangular box

    config = MACEConfig()
    config.r_max = 6.0
    config.hidden_irreps = '32x0e + 32x1o'
    config.batch_size = 4
    config.patience = 10
    model = MACEModel(config)

    model.add_atomic_energy('H', reference.compute_atomic_energy('H', box_size=6))
    model.add_atomic_energy('O', reference.compute_atomic_energy('O', box_size=6))
    model.add_atomic_energy('C', reference.compute_atomic_energy('C', box_size=6))
    model.add_atomic_energy('Al', reference.compute_atomic_energy('Al', box_size=6))

    # set learning parameters and do pretraining
    learning = SequentialLearning(
            path_output=path_output,
            niterations=10,
            train_valid_split=0.9,
            train_from_scratch=True,
            metrics=Metrics('MOF_phase_transition', 'examples', 'psiflow'),
            error_thresholds_for_reset=(10, 200), # in meV/atom, meV/angstrom
            initial_temperature=100,
            final_temperature=650,
            )

    # construct walkers; biased MTD MD in this case
    walkers = BiasedDynamicWalker.multiply(
            30,
            data_start=Dataset([atoms]),
            bias=bias,
            timestep=0.5,
            steps=1000,
            step=50,
            start=0,
            temperature=100,
            temperature_reset_quantile=0.01, # reset if P(temp) < 0.01
            pressure=0,
            )
    data = learning.run(
            model=model,
            reference=reference,
            walkers=walkers,
            )


def restart(path_output):
    reference = get_reference()
    learning  = load_learning(path_output)
    model, walkers, data_train, data_valid = load_state(path_output, '5')
    data_train, data_valid = learning.run(
            model=model,
            reference=reference,
            walkers=walkers,
            initial_data=data_train + data_valid,
            )


if __name__ == '__main__':
    psiflow.load()
    path_output = Path.cwd() / 'output' # stores learning results
    main(path_output)
    #restart(path_output)
