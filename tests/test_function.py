import numpy as np
from ase.units import kJ, mol

from psiflow.function import EinsteinCrystalFunction
from psiflow.hamiltonian import EinsteinCrystal


def test_einstein_crystal(dataset):
    function = EinsteinCrystalFunction(
        force_constant=1,
        centers=dataset[0].result().per_atom.positions,
        volume=0.0,
    )

    nstates = 4
    geometries = dataset[:nstates].reset().geometries().result()
    energy, forces = EinsteinCrystalFunction.apply(
        geometries,
        outputs=['value', 'grad_pos'],
        **function.parameters(),
    )
    assert np.all(energy >= 0)
    assert energy[0] == 0
    assert np.allclose(  # forces point to centers
        forces,
        function.centers.reshape(1, -1, 3) - dataset[:4].get('positions').result(),
    )
    assert geometries[0].energy is None
    forces = EinsteinCrystalFunction.apply(
        geometries,
        outputs=['grad_pos'],
        **function.parameters(),
    )[0]
    # for i, geometry in enumerate(geometries):
    #     assert np.allclose(
    #         geometry.per_atom.forces,
    #         forces[i, :len(geometry)],
    #     )

    energy_, forces_, stress_ = function.compute(dataset[:4])
    assert np.allclose(
        energy_.result(),
        energy,
    )
    assert np.allclose(
        forces_.result(),
        forces,
    )

    stress__, forces__ = function.compute(
        dataset[:4].geometries(),
        outputs=['grad_cell', 'grad_pos'],
    )
    assert np.allclose(
        stress__.result(),
        stress_.result(),
    )
    assert np.allclose(
        forces__.result(),
        forces_.result(),
    )

    forces = function.compute(dataset[:10], batch_size=3, outputs=['grad_pos'])
    forces_ = function.compute(dataset[:10].geometries(), batch_size=4, outputs=['grad_pos'])
    assert np.allclose(
        forces.result(),
        forces_.result(),
    )

    geometries = [dataset[i] for i in range(10)]
    forces__, energy = function.compute(geometries, batch_size=12, outputs=['grad_pos', 'value'])
    assert np.allclose(
        forces__.result(),
        forces.result(),
    )

    # hamiltonians can be evaluated to insert results in geometries
    hamiltonian = EinsteinCrystal(
        force_constant=function.force_constant,
        centers=function.centers,
    )
    data = dataset[:10].reset()
    evaluated = data.evaluate(hamiltonian, outputs=['energy'], batch_size=None)
    evaluated_ = data.evaluate(hamiltonian, outputs=['energy', 'forces'], batch_size=3)
    for i, geometry in enumerate(evaluated.geometries().result()):
        assert np.all(np.isnan(geometry.per_atom.forces))
        assert np.allclose(geometry.energy, energy.result()[i])
        assert np.allclose(
            evaluated_[i].result().energy,
            geometry.energy,
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
