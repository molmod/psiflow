import argparse
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
from psiflow.models import AllegroModel, AllegroConfig, MACEModel, MACEConfig
from psiflow.reference import CP2KReference
from psiflow.data import FlowAtoms, Dataset
from psiflow.sampling import RandomWalker, DynamicWalker, PlumedBias
from psiflow.ensemble import Ensemble


def get_allegro_model(context):
    AllegroModel.create_apps(context)
    config = AllegroConfig()
    config.loss_coeffs['total_energy'][0] = 10
    return AllegroModel(context, config)


def get_mace_model(context):
    MACEModel.create_apps(context)
    config = MACEConfig()
    config.max_num_epochs = 1000
    return MACEModel(context, config)


def main(context):
    """Simple training based on existing data"""
    train = Dataset.load(context, 'data/Al_mil53_train.xyz')
    valid = Dataset.load(context, 'data/Al_mil53_valid.xyz')
    model = get_mace_model(context)
    model.initialize(train)
    model.train(train, valid)
    model.deploy()
    errors = Dataset.get_errors(
            valid,
            model.evaluate(valid),
            )
    errors = np.mean(errors.result(), axis=0)
    print('energy error [RMSE, meV/atom]: {}'.format(errors[0]))
    print('forces error [RMSE, meV/A]   : {}'.format(errors[1]))
    print('stress error [RMSE, MPa]     : {}'.format(errors[2]))


if __name__ == '__main__':
    args = psiflow.experiment.parse_arguments()
    context, _ = psiflow.experiment.initialize(args)

    assert not args.restart
    main(context)
