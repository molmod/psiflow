import shutil
import requests
import logging
import yaml
from pathlib import Path
import numpy as np
import time

from ase.io import read

import psiflow.experiment
from psiflow.learning import OnlineLearning
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


def main(context, flow_manager):
    reference = get_reference(context) # CP2K; PBE-D3(BJ); TZVP
    model = get_mace_model(context) # NequIP; medium-sized network
    bias  = get_bias(context)
    data = Dataset.load(context, Path.cwd() / 'data' / 'Al_mil53_train.xyz')
    data_train = data[:40]   # start with minimal amount of data
    data_valid = data[40:50]

    # set learning parameters
    learning = OnlineLearning(
            train_valid_split=0.9,
            retrain_threshold=5,
            )

    # construct online learning ensemble; pure MD in this case
    walker = DynamicWalker(
            context,
            data[0],
            timestep=0.5,
            steps=100,
            step=50,
            start=0,
            temperature=600,
            pressure=0, # NPT
            force_threshold=20,
            initial_temperature=600,
            )
    ensemble = Ensemble.from_walker(
            walker,
            nwalkers=30, # 30 parallel walkers
            dataset=data_train, # used to initialize walkers
            )
    ensemble.add_bias(bias) # add separate MTD for every walker
    data_train, data_valid = learning.run(
            flow_manager=flow_manager,
            model=model,
            reference=reference,
            ensemble=ensemble,
            data_train=data_train,
            data_valid=data_valid,
            )


#def restart(context, flow_manager, restart_arg):
#    reference = get_reference(context)
#    model, ensemble, data_train, data_valid, checks = flow_manager.load(restart_arg, context)
#    learning = OnlineLearning(
#            niterations=5,
#            nstates=30,
#            retrain_model_per_iteration=True,
#            train_valid_split=0.9
#            )
#    data_train, data_valid = learning.run(
#            flow_manager=flow_manager,
#            model=model,
#            reference=reference,
#            ensemble=ensemble,
#            data_train=data_train,
#            data_valid=data_valid,
#            checks=checks,
#            )


if __name__ == '__main__':
    args = psiflow.experiment.parse_arguments()
    context, flow_manager = psiflow.experiment.initialize(args)

    if not args.restart:
        main(context, flow_manager)
    else:
        print('restarting from iteration {}'.format(args.restart))
        restart(context, flow_manager, args.restart)
