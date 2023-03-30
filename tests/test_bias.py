import requests
import pytest
import yaml
import numpy as np
import tempfile

from ase.build import bulk

from psiflow.data import Dataset
from psiflow.models import NequIPModel
from psiflow.sampling import RandomWalker, PlumedBias
from psiflow.sampling.bias import set_path_in_plumed, parse_plumed_input, \
        generate_external_grid, remove_comments_printflush


def test_get_filename_hills(tmp_path):
    plumed_input = """
#METAD COMMENT TO BE REMOVED
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV0: CV
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld
PRINT ARG=CV,metad.bias STRIDE=10 FILE=COLVAR
FLUSH STRIDE=10
"""
    plumed_input = remove_comments_printflush(plumed_input)
    plumed_input = set_path_in_plumed(plumed_input, 'METAD', '/tmp/my_input')
    plumed_input = set_path_in_plumed(plumed_input, 'METADD', '/tmp/my_input')
    assert plumed_input == """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV0: CV
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld FILE=/tmp/my_input
"""


def test_parse_plumed_input():
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
METAD ARG=CV1 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
"""
    with pytest.raises(AssertionError): # only one METAD allowed
        parse_plumed_input(plumed_input)
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
METAD ARG=CV2 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
RESTRAINT ARG=CV1 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
"""
    components, variables = parse_plumed_input(plumed_input)
    assert len(components) == 2
    assert len(variables) == 2
    assert variables[0] == 'CV1' # sorted alphabetically
    plumed_input = """
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
METADD ARG=CV1 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld FILE=/tmp/my_input
"""
    components, variables = parse_plumed_input(plumed_input)
    assert len(variables) == 1 # METADD is ignored as it is not known
    assert len(components) == 1
    assert components[0] == ('METAD', ('CV0',))


def test_bias_from_file(context, tmp_path):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV1: VOLUME
METAD ARG=CV1 SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
"""
    path_plumed = tmp_path / 'plumed_input.txt'
    with open(path_plumed, 'w') as f:
        f.write(plumed_input)
    bias  = PlumedBias.from_file(path_plumed)
    bias_ = PlumedBias(plumed_input)

    assert bias.plumed_input == bias_.plumed_input # check equivalence
    assert tuple(bias.data_futures.keys()) == tuple(bias_.data_futures.keys())


def test_bias_evaluate(context, dataset):
    kwargs = {
            'amplitude_box': 0.1,
            'amplitude_pos': 0.1,
            'seed': 0,
            }
    walker = RandomWalker(dataset[0], **kwargs)
    states = [walker.propagate() for i in range(10)]
    dataset = Dataset(states)

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV1: VOLUME
METAD ARG=CV1 SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
"""
    bias = PlumedBias(plumed_input)
    assert len(bias.components) == 1
    values = bias.evaluate(dataset).result()
    for i in range(dataset.length().result()):
        volume = np.linalg.det(dataset[i].result().cell)
        assert np.allclose(volume, values[i, 0])
    assert np.allclose(np.zeros(values[:, 1].shape), values[:, 1])
    dataset_ = bias.evaluate(dataset, as_dataset=True)
    for i, atoms in enumerate(dataset_.as_list()):
        assert np.allclose(atoms.get_volume(), atoms.info['CV1'])

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
"""
    bias = PlumedBias(plumed_input)
    values = bias.evaluate(dataset).result()
    assert np.allclose(
            values[:, 1],
            0.5 * (values[:, 0] - 150) ** 2,
            )
    singleton = Dataset(atoms_list=[dataset[0]])
    values_ = bias.evaluate(singleton).result()
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
    bias = PlumedBias(plumed_input, data)
    values = bias.evaluate(dataset).result()
    for i in range(dataset.length().result()):
        volume = np.linalg.det(dataset[i].result().cell)
        assert np.allclose(volume, values[i, 0])
    assert np.allclose(bias_function(values[:, 0]), values[:, 1])
    input_future, data_futures = bias.save(tmp_path)
    bias_ = PlumedBias.load(tmp_path)
    values_ = bias_.evaluate(dataset).result()
    assert np.allclose(values, values_)

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
external: EXTERNAL ARG=CV FILE=test_grid
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
"""
    bias = PlumedBias(plumed_input, data)
    assert len(bias.components) == 3
    values = bias.evaluate(dataset).result()
    for i in range(dataset.length().result()):
        volume = np.linalg.det(dataset[i].result().cell)
        assert np.allclose(volume, values[i, 0])
    reference = bias_function(values[:, 0]) + 0.5 * (values[:, 0] - 150) ** 2
    assert np.allclose(reference, values[:, 1])
    bias.save(tmp_path)
    bias_ = PlumedBias.load(tmp_path)
    values_ = bias_.evaluate(dataset).result()
    assert np.allclose(values, values_)


def test_adjust_restraint(context, dataset):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CVbla: MATHEVAL ARG=CV VAR=a FUNC=a*a PERIODIC=NO
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
RESTRAINT ARG=CVbla AT=150 KAPPA=0 LABEL=restraintbla
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
"""
    bias   = PlumedBias(plumed_input, data={})
    values = bias.evaluate(dataset).result()
    bias.adjust_restraint('CV', kappa=2, center=150)
    assert bias.plumed_input == """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CVbla: MATHEVAL ARG=CV VAR=a FUNC=a*a PERIODIC=NO
RESTRAINT ARG=CV AT=150 KAPPA=2 LABEL=restraint
RESTRAINT ARG=CVbla AT=150 KAPPA=0 LABEL=restraintbla
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
"""
    values_ = bias.evaluate(dataset).result()
    assert np.allclose(
            values[:, 0],
            values_[:, 0],
            )
    assert np.allclose(
            2 * values[:, 2],
            values_[:, 2],
            )
    bias.adjust_restraint('CV', kappa=3, center=155)
    assert bias.plumed_input == """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CVbla: MATHEVAL ARG=CV VAR=a FUNC=a*a PERIODIC=NO
RESTRAINT ARG=CV AT=155 KAPPA=3 LABEL=restraint
RESTRAINT ARG=CVbla AT=150 KAPPA=0 LABEL=restraintbla
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
"""
    values  = bias.evaluate(dataset, variable='CV').result()
    values_ = bias.evaluate(dataset, variable='CVbla').result()
    assert np.allclose(
            values ** 2,
            values_,
            )


def test_bias_find_states(context, dataset):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV0: VOLUME
RESTRAINT ARG=CV0 AT=150 KAPPA=1 LABEL=restraint
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=test_hills
CV1: DISTANCE ATOMS=1,2 COMPONENTS
RESTRAINT ARG=CV1.x AT=1 KAPPA=1 LABEL=bla
"""
    bias = PlumedBias(plumed_input, data={})
    values = bias.evaluate(dataset).result()
    assert values.shape == (dataset.length().result(), 3)
    values = values[:, 0]
    targets = np.array([np.min(values), np.mean(values), np.max(values)])
    extracted = bias.extract_grid(
            dataset,
            variable='CV0',
            targets=targets,
            )
    assert np.allclose(
            extracted[0].result().positions,
            dataset[int(np.argmin(values))].result().positions,
            )
    assert np.allclose(
            extracted[-1].result().positions,
            dataset[int(np.argmax(values))].result().positions,
            )
    values = bias.evaluate(extracted).result()[:, 0]
    assert np.allclose(
            values,
            np.sort(values),
            )
    extracted = bias.extract_between(
            dataset,
            variable='CV1.x',
            min_value=-1e10,
            max_value=1e10,
            )
    assert extracted.length().result() == dataset.length().result()
