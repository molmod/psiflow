import argparse
import shutil
import requests
import logging
import yaml
from pathlib import Path
import numpy as np
import time

from ase.io import read

import psiflow
from psiflow.learning import RandomLearning, OnlineLearning
from psiflow.models import NequIPModel, NequIPConfig, MACEModel, MACEConfig
from psiflow.reference import CP2KReference
from psiflow.data import FlowAtoms, Dataset
from psiflow.sampling import RandomWalker, DynamicWalker, PlumedBias
from psiflow.ensemble import Ensemble


def get_bias(context):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=50 HEIGHT=5 PACE=25 LABEL=metad FILE=test_hills
"""
    return PlumedBias(context, plumed_input)


def get_reference(context):
    CP2KReference.create_apps(context)
    with open(Path.cwd() / 'data' / 'cp2k_input.txt', 'r') as f:
        cp2k_input = f.read()
    reference = CP2KReference(context, cp2k_input=cp2k_input)
    basis     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/BASIS_MOLOPT_UZH').text
    dftd3     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/dftd3.dat').text
    potential = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/POTENTIAL_UZH').text
    cp2k_data = {
            'basis_set': basis,
            'potential': potential,
            'dftd3': dftd3,
            }
    for key, value in cp2k_data.items():
        with open(context.path / key, 'w') as f:
            f.write(value)
        reference.add_file(key, context.path / key)
    return reference


def get_nequip_model(context):
    NequIPModel.create_apps(context)
    config = NequIPConfig()
    config.loss_coeffs['total_energy'][0] = 10
    return NequIPModel(context, config)


def get_mace_model(context):
    MACEModel.create_apps(context)
    config = MACEConfig()
    config.max_num_epochs = 1000
    return MACEModel(context, config)


def main(context, flow_logger):
    reference = get_reference(context) # CP2K; PBE-D3(BJ); TZVP
    atoms = read(Path.cwd() / 'data' / 'Al_mil53_train.xyz')
    model = get_mace_model(context) # NequIP; medium-sized network
    bias  = get_bias(context)

    # FIRST STAGE: generate initial data by applying random perturbations
    walker = RandomWalker(
            context,
            atoms,
            amplitude_pos=0.08,
            amplitude_box=0.1,
            seed=0,
            )
    learning = RandomLearning(nstates=50, train_valid_split=0.9)
    data_train, data_valid = learning.run(
            flow_logger=flow_logger,
            model=model,
            reference=reference,
            walker=walker,
            bias=bias, # only used for logging errors in terms of CV
            )

    # SECOND STAGE: define MD walkers and simulate, evaluate, train
    walker = DynamicWalker( # basic MD parameters
            context,
            atoms,
            timestep=0.5,
            steps=300,
            step=50,
            start=0,
            temperature=600,
            pressure=None, # NVT
            force_threshold=20,
            initial_temperature=600,
            seed=0,
            )
    ensemble = Ensemble.from_walker(walker, nwalkers=30) # 30 parallel walkers
    ensemble.add_bias(bias) # add separate MTD for every walker
    learning = OnlineLearning(niterations=5, nstates=30)
    data_train, data_valid = learning.run(
            flow_logger=flow_logger,
            model=model,
            reference=reference,
            ensemble=ensemble,
            data_train=data_train,
            data_valid=data_valid,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--psiflow-config', action='store')
    parser.add_argument('--name', action='store', default=None)
    args = parser.parse_args()

    # initialize parsl, create directories
    context, flow_logger = psiflow.experiment.initialize(
            args.psiflow_config, # path to psiflow config.py
            args.name, # run name
            )
    main(context, flow_logger)
