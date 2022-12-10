import requests
import yaml
import numpy as np

from ase.build import bulk

from flower import Dataset
from flower.models import NequIPModel
from flower.sampling import DynamicWalker, RandomWalker, Ensemble, Bias
from flower.sampling.utils import set_path_hills_plumed, get_bias_plumed

from common import context, nequip_config
from test_dataset import dataset


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


def test_dynamic_walker_bias(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config, dataset)
    model.deploy()
    kwargs = {
            'timestep'           : 1,
            'steps'              : 10,
            'step'               : 1,
            'start'              : 0,
            'temperature'        : 100,
            'initial_temperature': 100,
            'pressure'           : None,
            }
    walker = DynamicWalker(context, dataset[0], **kwargs)

    # initial unit cell volume is around 125 A**3
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
restraint: RESTRAINT ARG=CV AT=150 KAPPA=1
"""
    bias = Bias(context, plumed_input)
    assert bias.kind[0] == 'RESTRAINT'
    assert bias.kind[1] == 'CV'
    walker.propagate(model=model, bias=bias)
    assert walker.tag_future.result() == 'safe'
    assert not np.allclose(
            walker.state_future.result().positions,
            walker.start_future.result().positions,
            )

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
""" # RESTART automatically added in input if not present
    bias = Bias(context, plumed_input)
    assert bias.kind[0] == 'METAD'
    assert bias.kind[1] == 'CV'
    kwargs = {
            'timestep'           : 1,
            'steps'              : 10,
            'step'               : 1,
            'start'              : 0,
            'temperature'        : 100,
            'initial_temperature': 100,
            'pressure'           : None,
            }
    walker.propagate(model=model, bias=bias)
    assert bias.hills_future is not None

    with open(bias.hills_future.result(), 'r') as f:
        single_length = len(f.read().split('\n'))
    assert walker.tag_future.result() == 'safe'
    walker.propagate(model=model, bias=bias)
    with open(bias.hills_future.result(), 'r') as f:
        double_length = len(f.read().split('\n'))
    assert double_length == 2 * single_length - 1 # twice as many gaussians


def test_bias_evaluate(context, dataset):
    kwargs = {
            'amplitude_box': 0.1,
            'amplitude_pos': 0.1,
            'seed': 0,
            }
    walker = RandomWalker(context, dataset[0], **kwargs)
    ensemble = Ensemble.from_walker(walker, nwalkers=10)
    dataset = ensemble.propagate()

    plumed_input = """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
FLUSH STRIDE=1
"""
    bias = Bias(context, plumed_input)
    values = bias.evaluate(dataset).result()
    for i in range(dataset.length().result()):
        volume = np.linalg.det(dataset[i].result().cell)
        assert np.allclose(volume, values[i, 0])
    assert np.allclose(np.zeros(values[:, 1].shape), values[:, 1])

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
"""
    bias = Bias(context, plumed_input)
    values = bias.evaluate(dataset).result()
    assert np.allclose(
            values[:, 1],
            0.5 * (values[:, 0] - 150) ** 2,
            )
