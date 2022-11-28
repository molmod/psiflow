import pytest
import yaml
import requests
import numpy as np

from autolearn import ModelExecution
from autolearn.sampling import DynamicWalker
from autolearn.sampling.utils import ForceThresholdExceededException
from autolearn.dataset import Sample, Dataset
from autolearn.models import NequIPModel
from autolearn.base import ModelExecution

from utils import generate_emt_cu_data


def test_dynamic_walker_sampling(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['root'] = str(tmp_path)

    training = Dataset.from_atoms_list(
            generate_emt_cu_data(a=5, nstates=5),
            )
    model = NequIPModel(config)
    model.initialize(training)
    model_execution = ModelExecution()

    kwargs = {
            'timestep'           : 1,
            'steps'              : 5,
            'step'               : 1,
            'start'              : 0,
            'temperature'        : 300,
            'pressure'           : None,
            }
    walker = DynamicWalker(training[0], **kwargs)
    walker = walker.propagate(model, model_execution)
    sample = walker.sample()

    kwargs = {
            'timestep'           : 1,
            'steps'              : 5,
            'step'               : 1,
            'start'              : 0,
            'temperature'        : 300,
            'pressure'           : None,
            'force_threshold'    : 0.0001, # exception should be raised immediately
            }
    walker = DynamicWalker(training[0], **kwargs)
    with pytest.raises(ForceThresholdExceededException):
        walker = walker.propagate(model, model_execution)
