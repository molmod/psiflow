import requests
import yaml
import numpy as np
import tempfile

from ase.build import bulk

from flower.data import Dataset
from flower.models import NequIPModel
from flower.sampling import DynamicWalker, RandomWalker, PlumedBias
from flower.sampling.bias import set_path_in_plumed, parse_plumed_input, \
        generate_external_grid
from flower.ensemble import Ensemble


def test_get_filename_hills(tmp_path):
    plumed_input = """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld
PRINT ARG=CV,metad.bias STRIDE=10 FILE=COLVAR
FLUSH STRIDE=10
"""
    plumed_input = set_path_in_plumed(plumed_input, 'METAD', '/tmp/my_input')
    plumed_input = set_path_in_plumed(plumed_input, 'METADD', '/tmp/my_input')
    assert plumed_input == """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld FILE=/tmp/my_input
PRINT ARG=CV,metad.bias STRIDE=10 FILE=COLVAR
FLUSH STRIDE=10
"""
    assert parse_plumed_input(plumed_input)[0] == ('METAD', 'CV')


def test_dynamic_walker_bias(context, nequip_config, dataset):
    model = NequIPModel(context, nequip_config)
    model.initialize(dataset)
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
    bias = PlumedBias(context, plumed_input)
    assert bias.components[0] == ('RESTRAINT', 'CV')
    walker.propagate(model=model, bias=bias)
    assert walker.tag_future.result() == 'safe'
    assert not np.allclose(
            walker.state_future.result().positions,
            walker.start_future.result().positions,
            )

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
restraint: RESTRAINT ARG=CV AT=150 KAPPA=1
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
""" # RESTART automatically added in input if not present
    bias = PlumedBias(context, plumed_input)
    assert ('METAD', 'CV') in bias.components
    assert ('RESTRAINT', 'CV') in bias.components
    assert bias.keys[0] == 'METAD'
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

    with open(bias.data_futures['METAD'].result(), 'r') as f:
        single_length = len(f.read().split('\n'))
    assert walker.tag_future.result() == 'safe'
    walker.propagate(model=model, bias=bias)
    with open(bias.data_futures['METAD'].result(), 'r') as f:
        double_length = len(f.read().split('\n'))
    assert double_length == 2 * single_length - 1 # twice as many gaussians

    # double check MTD gives correct nonzero positive contribution
    values = bias.evaluate(dataset, variable='CV').result()
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=2 LABEL=metad FILE=test_hills
""" # RESTART automatically added in input if not present
    bias_mtd = PlumedBias(context, plumed_input, data={'METAD': bias.data_futures['METAD']})
    values_mtd = bias_mtd.evaluate(dataset, variable='CV').result()
    assert np.allclose(
            values[:, 0],
            values_mtd[:, 0],
            )
    assert np.any(values_mtd[:, 1] >  0)
    assert np.all(values_mtd[:, 1] >= 0)
    total  = values[:, 1]
    manual = values_mtd[:, 1] + 0.5 * (values[:, 0] - 150) ** 2
    assert np.allclose(
            total,
            manual,
            )
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
SOMEOTHER: VOLUME
restraint: RESTRAINT ARG=SOMEOTHER AT=150 KAPPA=1
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
""" # RESTART automatically added in input if not present
    bias_ = PlumedBias(context, plumed_input, data={'METAD': bias.data_futures['METAD']})
    values_ = bias_.evaluate(dataset, variable='CV').result()
    assert np.allclose(
            values_[:, 0],
            values[:, 0],
            )
    assert np.allclose( # even though restraint is present, it should not count
            values_[:, 1],
            values_mtd[:, 1],
            )


def test_bias_evaluate(context, dataset):
    kwargs = {
            'amplitude_box': 0.1,
            'amplitude_pos': 0.1,
            'seed': 0,
            }
    walker = RandomWalker(context, dataset[0], **kwargs)
    ensemble = Ensemble.from_walker(walker, nwalkers=10)
    dataset = ensemble.sample(nstates=10)

    plumed_input = """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
FLUSH STRIDE=1
"""
    bias = PlumedBias(context, plumed_input)
    assert len(bias.components) == 1
    assert tuple(bias.keys) == ('METAD',)
    values = bias.evaluate(dataset, variable='CV').result()
    for i in range(dataset.length().result()):
        volume = np.linalg.det(dataset[i].result().cell)
        assert np.allclose(volume, values[i, 0])
    assert np.allclose(np.zeros(values[:, 1].shape), values[:, 1])

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
"""
    bias = PlumedBias(context, plumed_input)
    values = bias.evaluate(dataset, variable='CV').result()
    assert np.allclose(
            values[:, 1],
            0.5 * (values[:, 0] - 150) ** 2,
            )
    singleton = Dataset(context, atoms_list=[dataset[0]])
    values_ = bias.evaluate(singleton, variable='CV').result()
    assert np.allclose(
            values[0, :],
            values_[0, :],
            )


def test_bias_external(context, dataset, tmp_path):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
external: EXTERNAL ARG=CV FILE=test_grid
"""
    bias_function = lambda x: np.exp(-0.01 * (x - 150) ** 2)
    variable = np.linspace(0, 300, 500)
    grid = generate_external_grid(bias_function, variable, 'CV', periodic=False)
    data = {'EXTERNAL': grid}
    bias = PlumedBias(context, plumed_input, data)
    values = bias.evaluate(dataset, variable='CV').result()
    for i in range(dataset.length().result()):
        volume = np.linalg.det(dataset[i].result().cell)
        assert np.allclose(volume, values[i, 0])
    assert np.allclose(bias_function(values[:, 0]), values[:, 1])
    input_future, data_futures = bias.save(tmp_path)
    bias_ = PlumedBias.load(context, tmp_path)
    values_ = bias_.evaluate(dataset, variable='CV').result()
    assert np.allclose(values, values_)

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
external: EXTERNAL ARG=CV FILE=test_grid
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
"""
    bias = PlumedBias(context, plumed_input, data)
    assert len(bias.components) == 3
    assert bias.keys[0] == 'METAD' # in front
    values = bias.evaluate(dataset, variable='CV').result()
    for i in range(dataset.length().result()):
        volume = np.linalg.det(dataset[i].result().cell)
        assert np.allclose(volume, values[i, 0])
    reference = bias_function(values[:, 0]) + 0.5 * (values[:, 0] - 150) ** 2
    assert np.allclose(reference, values[:, 1])
    bias.save(tmp_path)
    bias_ = PlumedBias.load(context, tmp_path)
    values_ = bias_.evaluate(dataset, variable='CV').result()
    assert np.allclose(values, values_)


def test_adjust_restraint(context, dataset):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
"""
    bias   = PlumedBias(context, plumed_input, data={})
    values = bias.evaluate(dataset, variable='CV').result()
    bias.adjust_restraint('CV', kappa=2, center=150)
    assert bias.plumed_input == """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=2 LABEL=restraint
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
"""
    values_ = bias.evaluate(dataset, variable='CV').result()
    assert np.allclose(
            values[:, 0],
            values_[:, 0],
            )
    assert np.allclose(
            2 * values[:, 1],
            values_[:, 1],
            )
    bias.adjust_restraint('CV', kappa=3, center=155)
    assert bias.plumed_input == """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=155 KAPPA=3 LABEL=restraint
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
"""


def test_bias_find_states(context, dataset):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
"""
    bias = PlumedBias(context, plumed_input, data={})
    values = bias.evaluate(dataset, variable='CV').result()[:, 0]
    #cv_min = np.min(cv_values)
    #cv_max = np.max(cv_values)
    #cv_step = (cv_max - cv_min) / 2
    targets = np.array([np.min(values), np.mean(values), np.max(values)])
    extracted = bias.extract_states(
            dataset,
            variable='CV',
            targets=targets,
            slack=np.max(values) - np.min(values), # disable slack
            )
    assert np.allclose(
            extracted[0].result().positions,
            dataset[int(np.argmin(values))].result().positions,
            )
    assert np.allclose(
            extracted[-1].result().positions,
            dataset[int(np.argmax(values))].result().positions,
            )
    values = bias.evaluate(extracted, variable='CV').result()[:, 0]
    assert np.allclose(
            values,
            np.sort(values),
            )
