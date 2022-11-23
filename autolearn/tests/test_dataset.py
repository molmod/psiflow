import requests
import yaml
import numpy as np

#import covalent as ct
#c = ct.get_config() # avoid creating a results dir each time tests are executed
#c['dispatcher']['results_dir'] = './'
#ct.set_config(c)

from autolearn import Dataset, ModelExecution
from autolearn.base import BaseModel
from autolearn.models import NequIPModel

from test_models import generate_dummy_data
