import requests
import yaml
import numpy as np

from ase.build import bulk

from autolearn import Dataset, ModelExecution, Bias, Sample
from autolearn.models import NequIPModel
from autolearn.sampling import DynamicWalker, RandomWalker, Ensemble
from autolearn.utils import set_path_hills_plumed, get_bias_plumed

from utils import generate_emt_cu_data


def test_get_filename_hills(tmp_path):
    plumed_input = """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills sdld
PRINT ARG=CV,metad.bias STRIDE=10 FILE=COLVAR
FLUSH STRIDE=10
"""
    plumed_input = set_path_hills_plumed(plumed_input, '/tmp/my_input')
    assert plumed_input == """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
PRINT ARG=CV,metad.bias STRIDE=10 FILE=COLVAR
FLUSH STRIDE=10
"""
    assert get_bias_plumed(plumed_input) == ('METAD', 'CV')


def test_dynamic_walker_bias(tmp_path):
    config_text = requests.get('https://raw.githubusercontent.com/mir-group/nequip/v0.5.5/configs/minimal.yaml').text
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    config['root'] = str(tmp_path)
    # ensure stress is computed by the model
    config['model_builders'] = ['SimpleIrrepsConfig', 'EnergyModel',
            'PerSpeciesRescale', 'StressForceOutput', 'RescaleEnergyEtc']

    training = Dataset.from_atoms_list(
            generate_emt_cu_data(a=5, nstates=5),
            )
    model = NequIPModel(config)
    model.initialize(training)
    model_execution = ModelExecution()

    # initial unit cell volume is around 125 A**3
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
restraint: RESTRAINT ARG=CV AT=150 KAPPA=1
"""
    bias = Bias(plumed_input)
    assert not bias.uses_hills
    kwargs = {
            'timestep'           : 1,
            'steps'              : 10,
            'step'               : 1,
            'start'              : 0,
            'temperature'        : 100,
            'initial_temperature': 100,
            'pressure'           : 0,
            'bias'               : bias,
            }
    walker = DynamicWalker(training[0], **kwargs)
    walker = walker.propagate(model, model_execution)

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
""" # RESTART automatically added in input if not present
    bias = Bias(plumed_input)
    kwargs = {
            'timestep'           : 1,
            'steps'              : 10,
            'step'               : 1,
            'start'              : 0,
            'temperature'        : 100,
            'initial_temperature': 100,
            'pressure'           : 0,
            'bias'               : bias,
            }
    walker = DynamicWalker(training[0], **kwargs)
    walker = walker.propagate(model, model_execution)
    assert walker.parameters.bias.hills is not None
    single_length = len(bias.hills.split('\n'))
    walker = walker.propagate(model, model_execution)
    double_length = len(bias.hills.split('\n'))
    assert double_length == 2 * single_length - 1 # twice as many gaussians


def test_bias_evaluate(tmp_path):
    sample = Sample(bulk('Cu', 'fcc', a=10, cubic=True))
    kwargs = {
            'amplitude_box': 0.1,
            'amplitude_pos': 0.1,
            'seed': 0,
            }
    walker = RandomWalker(sample, **kwargs)
    ensemble = Ensemble.from_walker(walker, nwalkers=10)
    ensemble.propagate(None, None)
    dataset = ensemble.sample()

    plumed_input = """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
FLUSH STRIDE=1
"""
    bias = Bias(plumed_input)
    values = bias.evaluate(dataset)
    for i, sample in enumerate(dataset):
        volume = np.linalg.det(sample.atoms.cell)
        assert np.allclose(volume, values[i, 0])
    assert np.allclose(np.zeros(values[:, 1].shape), values[:, 1])
