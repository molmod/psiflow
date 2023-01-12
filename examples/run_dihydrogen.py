import argparse
import shutil
import requests
import yaml
from pathlib import Path
import numpy as np

import parsl
from parsl.utils import get_all_checkpoints

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
    if not args.restart:
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
    config.checkpoint_mode = 'task_exit'
    if args.restart:
        config.checkpoint_files = get_all_checkpoints(str(path_internal))
        print('found {} checkpoint files'.format(len(config.checkpoint_files)))
    parsl.load(config)
    config.retries = args.retries
    context = ExecutionContext(config, path=path_context)
    context.register(ModelExecutionDefinition())
    context.register(ReferenceExecutionDefinition(time_per_singlepoint=30))
    context.register(TrainingExecutionDefinition())

    # setup manager for IO, wandb logging
    path_output  = path_run / 'output'
    manager = Manager(
            path_output,
            wandb_project='dihydrogen',
            wandb_group=args.name,
            restart=args.restart,
            error_x_axis='CV', # plot errors w.r.t CV value
            )
    return context, manager


def get_bias(context):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2
METAD ARG=CV SIGMA=0.05 HEIGHT=5 PACE=5 LABEL=metad FILE=test_hills
"""
#RESTRAINT ARG=CV AT=0.74 KAPPA=0 LABEL=restraint
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
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['r_max'] = 5.0 # reduce computational cost of data processing
    config['chemical_symbols'] = ['X'] # should get overridden
    config['num_layers'] = 2
    config['num_features'] = 8
    config['invariant_neurons'] = 16 #16
    config['invariant_layers'] = 1
    return NequIPModel(context, config)


def main(context, manager):
    atoms = FlowAtoms(
            numbers=np.ones(2),
            positions=np.array([[0, 0, 0], [0, 0, 0.8]]),
            cell=np.eye(3) * 6,
            pbc=True,
            )
    walker = RandomWalker(
            context,
            atoms,
            amplitude_pos=0.05,
            amplitude_box=0.01,
            seed=0,
            )
    reference = get_reference(context) # CP2K; PBE-D3(BJ); TZVP
    model = get_model(context) # NequIP; small network
    bias = get_bias(context)

    # initial stage: random perturbations
    learning = RandomLearning(nstates=5, train_valid_split=0.8)
    data_train, data_valid = learning.run(
            manager=manager,
            model=model,
            reference=reference,
            walker=walker,
            bias=bias,
            )

    # used biased dynamics to sample phase space
    walker = DynamicWalker(
            context,
            atoms,
            timestep=0.5,
            steps=10,
            step=1,
            start=0,
            temperature=600,
            pressure=None, # NVT
            force_threshold=30,
            initial_temperature=600,
            seed=0,
            )
    ensemble = Ensemble.from_walker(walker, nwalkers=4)
    ensemble.biases[:] = [bias.copy(), bias.copy(), bias.copy(), bias.copy()]
    learning = OnlineLearning(niterations=3, nstates=4)
    data_train, data_valid = learning.run(
            manager=manager,
            model=model,
            reference=reference,
            ensemble=ensemble,
            data_train=data_train,
            data_valid=data_valid,
            )
    evaluated = model.evaluate(data_train)
    for i in range(evaluated.length().result()):
        print(evaluated[i].result().info['energy'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsl-config', action='store')
    parser.add_argument('--name', action='store')
    parser.add_argument('--retries', action='store', default=1)
    parser.add_argument('--restart', action='store_true', default=False)
    args = parser.parse_args()

    context, manager = get_context_and_manager(args)
    main(context, manager)
