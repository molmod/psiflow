import requests
import yaml
import numpy as np

from ase import Atoms
from ase.build import bulk

from autolearn import Dataset, ModelExecution, Sample
from autolearn.models import NequIPModel
from autolearn.sampling import DynamicWalker, Ensemble, RandomWalker

from utils import generate_emt_cu_data


def test_ensemble_dynamic(tmp_path):
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
            'steps'              : 1,
            'step'               : 1,
            'start'              : 0,
            'temperature'        : 300,
            'pressure'           : None,
            'bias'               : None,
            }
    walker = DynamicWalker(training[0], **kwargs)
    ensemble = Ensemble.from_walker(walker, nwalkers=5)
    ensemble.propagate(model, model_execution)
    dataset = ensemble.sample()
    assert len(dataset) == 5

    # ensure all states are significantly different from each other due to
    # different initial seed (as set in Ensemble.from_walker)
    for i in range(4):
        for j in range(i + 1, 5):
            pos0 = dataset[i].atoms.get_positions()
            pos1 = dataset[j].atoms.get_positions()
            assert not np.allclose(pos0, pos1, rtol=1e-2)


def test_ensemble_random(tmp_path):
    sample = Sample(bulk('Cu', 'fcc', a=10, cubic=True))
    ensemble = Ensemble.from_walker(RandomWalker(sample), nwalkers=5)
    ensemble.propagate(model=None, model_execution=None)
    dataset = ensemble.sample()

    # ensure all states are significantly different from each other due to
    # different initial seed (as set in Ensemble.from_walker)
    for i in range(4):
        for j in range(i + 1, 5):
            pos0 = dataset[i].atoms.get_positions()
            pos1 = dataset[j].atoms.get_positions()
            assert not np.allclose(pos0, pos1)
