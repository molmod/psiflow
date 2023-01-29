import argparse
import shutil
import requests
import yaml
from pathlib import Path
import numpy as np

from ase.io import read

import parsl
from parsl.utils import get_all_checkpoints

from psiflow.manager import Manager
from psiflow.learning import RandomLearning, OnlineLearning
from psiflow.models import NequIPModel, MACEModel, MACEConfig
from psiflow.reference import CP2KReference
from psiflow.data import FlowAtoms
from psiflow.sampling import RandomWalker, DynamicWalker, PlumedBias
from psiflow.ensemble import Ensemble
from psiflow.execution import ExecutionContext
from psiflow.utils import get_parsl_config_from_file


def get_context_and_manager(args):
    path_run = Path.cwd() / args.name
    if args.restart is None:
        if path_run.is_dir():
            shutil.rmtree(path_run)
        path_run.mkdir()
    else:
        assert path_run.is_dir()
    path_internal = path_run / 'parsl_internal'
    path_context  = path_run / 'context_dir'
    config = get_parsl_config_from_file(
            args.parsl_config,
            path_internal,
            )
    config.initialize_logging = False
    parsl.load(config)
    config.retries = args.retries
    context = ExecutionContext(config, path=path_context)

    # setup manager for IO, wandb logging
    path_output  = path_run / 'output'
    manager = Manager(
            path_output,
            wandb_project='nanoporous',
            wandb_group=args.name,
            restart=(args.restart is not None),
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
    context.define_execution(
            CP2KReference,
            executor='reference',
            device='cpu',
            ncores=None,
            mpi_command=lambda x: f'mpirun -np {x} ',
            cp2k_exec='cp2k.psmp',
            time_per_singlepoint=20,
            )
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
    context.define_execution(
            NequIPModel,
            evaluate_executor='model',
            evaluate_device='cpu',
            evaluate_ncores=None,
            evaluate_dtype='float32',
            training_executor='training',
            training_device='cuda',
            training_ncores=None,
            training_dtype='float32',
            training_walltime=80,
            )
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


def get_mace_model(context):
    context.define_execution(
            MACEModel,
            evaluate_executor='model',
            evaluate_device='cpu',
            evaluate_ncores=None,
            evaluate_dtype='float32',
            training_executor='training',
            training_device='cuda',
            training_ncores=None,
            training_dtype='float32',
            training_walltime=3600,
            )
    config = MACEConfig()
    return MACEModel(context, config)


def main(context, manager, restart):
    reference = get_reference(context) # CP2K; PBE-D3(BJ); TZVP
    atoms = read(Path.cwd() / 'data' / 'Al_mil53.xyz')
    if restart is None: # generate initial data with random learning
        model = get_nequip_model(context) # NequIP; medium-sized network
        bias = get_bias(context)
        walker = RandomWalker(
                context,
                atoms,
                amplitude_pos=0.08,
                amplitude_box=0.1,
                seed=0,
                )
        # initial stage: random perturbations
        learning = RandomLearning(nstates=50, train_valid_split=0.9)
        data_train, data_valid = learning.run(
                manager=manager,
                model=model,
                reference=reference,
                walker=walker,
                bias=bias, # only there for wandb logging
                )
        walker = DynamicWalker( # MD
                context,
                atoms,
                timestep=0.5,
                steps=300,
                step=50,
                start=0,
                temperature=600,
                pressure=0, # NPT
                force_threshold=20,
                initial_temperature=600,
                seed=0,
                )
        ensemble = Ensemble.from_walker(walker, nwalkers=30)
        ensemble.add_bias(bias) # add separate MTD for every walker
    else:
        model, ensemble, data_train, data_valid, _ = manager.load(restart, context)
    learning = OnlineLearning(niterations=5, nstates=30)
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
    parser.add_argument('--retries', action='store', default=3)
    parser.add_argument('--restart', action='store', default=None)
    args = parser.parse_args()

    context, manager = get_context_and_manager(args)
    main(context, manager, restart=args.restart)
