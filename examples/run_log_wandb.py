import requests
import logging
from pathlib import Path
import numpy as np

import psiflow
from psiflow.models import NequIPModel, NequIPConfig, MACEModel, MACEConfig
from psiflow.data import Dataset
from psiflow.reference import CP2KReference
from psiflow.sampling import DynamicWalker, PlumedBias
from psiflow.wandb_utils import WandBLogger


def get_bias():
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=200 HEIGHT=5 PACE=100 LABEL=metad FILE=test_hills
"""
    return PlumedBias(plumed_input)


def get_mace_model():
    config = MACEConfig()
    config.max_num_epochs = 1000
    return MACEModel(config)


def main():
    train = Dataset.load('data/Al_mil53_train.xyz')
    train.data_future.result()
    valid = Dataset.load('data/Al_mil53_valid.xyz')
    bias  = get_bias()
    model = get_mace_model()
    model.initialize(train)
    model.deploy()

    wandb_logger = WandBLogger(
            wandb_project='psiflow',
            wandb_group='run_log_wandb',
            error_x_axis='CV',
            )
    log = wandb_logger('untrained', model, data_valid=bias.evaluate(valid, as_dataset=True))
    log.result()


if __name__ == '__main__':
    psiflow.load()
    main()
