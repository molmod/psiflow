import argparse
import shutil
import requests
import yaml
from pathlib import Path
import numpy as np

from ase.io import read

import parsl

from psiflow.manager import Manager
from psiflow.learning import RandomLearning, OnlineLearning
from psiflow.models import NequIPModel
from psiflow.reference import CP2KReference
from psiflow.data import FlowAtoms
from psiflow.sampling import RandomWalker, DynamicWalker, PlumedBias
from psiflow.ensemble import Ensemble
from psiflow.execution import ExecutionContext, ModelExecutionDefinition, \
        ReferenceExecutionDefinition, TrainingExecutionDefinition
from psiflow.utils import get_parsl_config_from_file


def get_context_and_manager(args):
    path_run = Path.cwd() / args.name
    if path_run.is_dir():
        shutil.rmtree(path_run)
    path_run.mkdir()
    path_internal = path_run / 'parsl_internal'
    path_context  = path_run / 'context_dir'
    config = get_parsl_config_from_file(
            args.parsl_config,
            path_internal,
            )
    parsl.load(config)
    config.retries = args.retries
    config.cache=True
    config.initialize_logging = False
    context = ExecutionContext(config, path=path_context)
    context.register(ModelExecutionDefinition())
    context.register(ReferenceExecutionDefinition(time_per_singlepoint=500))
    context.register(TrainingExecutionDefinition(walltime=3600))

    # setup manager for IO, wandb logging
    path_output  = path_run / 'output'
    manager = Manager(
            path_output,
            wandb_project='zeolite',
            wandb_group=args.name,
            error_x_axis='CV', # plot errors w.r.t CV value
            )
    return context, manager


def get_bias(context):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=50 HEIGHT=5 PACE=25 LABEL=metad FILE=test_hills
"""
    return PlumedBias(context, plumed_input)


def get_reference(context):
    with open(Path.cwd() / 'data' / 'cp2k_input.txt', 'r') as f:
        cp2k_input = f.read()
    basis     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/BASIS_MOLOPT_UZH').text
    dftd3     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/dftd3.dat').text
    potential = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/POTENTIAL_UZH').text
    cp2k_data = {
            'BASIS_SET_FILE_NAME': basis,
            'POTENTIAL_FILE_NAME': potential,
            'PARAMETER_FILE_NAME': dftd3,
            }
    return CP2KReference(context, cp2k_input=cp2k_input, cp2k_data=cp2k_data)


def get_model(context):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.6/configs/full.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['r_max'] = 5.0 # reduce computational cost of data processing
    config['chemical_symbols'] = ['X'] # should get overridden
    config['num_layers'] = 4
    config['num_features'] = 16
    config['l_max'] = 1
    config['loss_coeffs']['total_energy'][0] = 10 # increase energy weight
    config['loss_coeffs']['total_energy'][1] = 'PerAtomMSELoss'
    config['model_builders'][3] = 'StressForceOutput' # include stress in output
    return NequIPModel(context, config)


def main(context, manager):
    atoms = read(Path.cwd() / 'data' / 'zeolite.xyz')
    walker = RandomWalker(
            context,
            atoms,
            amplitude_pos=0.08,
            amplitude_box=0.08,
            seed=0,
            )
    reference = get_reference(context) # CP2K; PBE-D3(BJ); TZVP
    model = get_model(context) # NequIP; medium-sized network
    bias = get_bias(context)

    # initial stage: random perturbations
    learning = RandomLearning(nstates=20, train_valid_split=0.8)
    data_train, data_valid = learning.run(
            manager=manager,
            model=model,
            reference=reference,
            walker=walker,
            bias=bias, # only there for wandb logging
            )

    # used biased dynamics to sample phase space
    walker = DynamicWalker(
            context,
            atoms,
            timestep=0.5,
            steps=200,
            step=50,
            start=0,
            temperature=1000,
            pressure=0, # NPT
            force_threshold=40,
            initial_temperature=1000,
            seed=0,
            )
    ensemble = Ensemble.from_walker(walker, nwalkers=10)
    ensemble.add_bias(bias) # separate MTD for every walker
    learning = OnlineLearning(niterations=5, nstates=10)
    data_train, data_valid = learning.run(
            manager=manager,
            model=model,
            reference=reference,
            ensemble=ensemble,
            data_train=data_train,
            data_valid=data_valid,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsl-config', action='store')
    parser.add_argument('--name', action='store')
    parser.add_argument('--retries', action='store', default=1)
    args = parser.parse_args()

    context, manager = get_context_and_manager(args)
    main(context, manager)
