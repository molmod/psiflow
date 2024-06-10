import numpy as np
from ase.units import kJ, mol

from psiflow.functions import PlumedFunction, EinsteinCrystalFunction, \
        extend_with_future_support


def test_einstein_crystal(dataset):
    function = EinsteinCrystalFunction(
        force_constant=1,
        centers=dataset[0].result().per_atom.positions,
        volume=0.0,
    )

    outputs = function(dataset[:10].geometries().result())
    assert np.all(outputs['energy'] >= 0)
    assert outputs['energy'][0] == 0
    assert np.allclose(  # forces point to centers
        outputs['forces'],
        function.centers.reshape(1, -1, 3) - dataset[:10].get('positions').result(),
    )


def test_plumed_function(tmp_path, dataset, dataset_h2):
    data = dataset + dataset_h2
    plumed_str = """
D1: DISTANCE ATOMS=1,2 NOPBC
CV: BIASVALUE arg=D1
"""
    function = PlumedFunction(plumed_str)
    outputs = function(data.geometries().result())

    f = 1 / (kJ / mol) * 10  # eV --> kJ/mol and nm --> A
    positions = data.get('positions').result()
    manual = np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=1)
    assert np.allclose(
        outputs['energy'] * f,
        manual,
    )
    gradient = (positions[:, 0, :] - positions[:, 1, :]) / manual.reshape(-1, 1)
    assert np.allclose(
        outputs['forces'][:, 0, :] * f,
        gradient * (-1.0),
    )

    # use external grid as bias, check that file is read
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=50 KAPPA=1
"""
    function = PlumedFunction(plumed_input)
    outputs = function(dataset.geometries().result())

    volumes = np.linalg.det(dataset.get('cell').result())
    energy_ = (volumes - 50) ** 2 * (kJ / mol) / 2
    assert np.allclose(
        outputs['energy'],
        energy_,
    )


def test_future_support():
    extend_with_future_support(EinsteinCrystalFunction)
