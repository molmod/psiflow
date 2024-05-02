import copy
import json

import numpy as np
import pytest
from ase import Atoms
from ase.units import kJ, mol
from parsl.data_provider.files import File

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import NullState
from psiflow.hamiltonians import (
    EinsteinCrystal,
    Harmonic,
    MACEHamiltonian,
    PlumedHamiltonian,
    deserialize_calculator,
)
from psiflow.hamiltonians._plumed import remove_comments_printflush, set_path_in_plumed
from psiflow.hamiltonians.hamiltonian import MixtureHamiltonian, Zero
from psiflow.hamiltonians.utils import (
    ForceMagnitudeException,
    PlumedCalculator,
    check_forces,
)
from psiflow.utils import copy_app_future, copy_data_future, dump_json


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
    plumed_input = set_path_in_plumed(plumed_input, "METAD", "/tmp/my_input")
    plumed_input = set_path_in_plumed(plumed_input, "METADD", "/tmp/my_input")
    assert (
        plumed_input.strip()
        == """
RESTART
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV0: CV
METAD ARG=CV0 SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad FILE=/tmp/my_input sdld
METADD ARG=CV SIGMA=100 HEIGHT=2 PACE=50 LABEL=metad sdld FILE=/tmp/my_input
""".strip()
    )


def test_einstein(context, dataset, dataset_h2):
    state = dataset[0]
    hamiltonian = EinsteinCrystal(state, force_constant=1)
    evaluated = hamiltonian.evaluate(dataset[:10])
    assert evaluated[0].result().energy == 0.0
    for i in range(1, 10):
        assert evaluated[i].result().energy > 0.0
        assert not np.allclose(evaluated[i].result().stress, 0.0)

    # test evaluation with NullState in data
    data = hamiltonian.evaluate(dataset[:5] + Dataset([NullState]) + dataset[5:10])
    energies = data.get("energy").result()
    assert np.isnan(energies[5])

    # test nonperiodic evaluation
    einstein = EinsteinCrystal(dataset_h2[3], force_constant=0.1)
    data = einstein.evaluate(dataset_h2)
    energies, forces, stress = data.get("energy", "forces", "stress")
    assert np.all(energies.result() >= 0)
    assert np.any(energies.result() > 0)
    assert np.all(np.isnan(stress.result()))

    # test batched evaluation
    energies = np.array([evaluated[i].result().energy for i in range(10)])
    evaluated = hamiltonian.evaluate(dataset[:10], batch_size=3)
    for i in range(10):
        assert energies[i] == evaluated[i].result().energy

    # test equality
    hamiltonian_ = EinsteinCrystal(state.result(), force_constant=1.1)
    assert not hamiltonian == hamiltonian_
    hamiltonian_ = EinsteinCrystal(state, force_constant=1.0)
    assert hamiltonian != hamiltonian_  # app future copied
    hamiltonian_.reference_geometry = hamiltonian.reference_geometry
    assert hamiltonian == hamiltonian_
    hamiltonian_ = EinsteinCrystal(dataset[1], force_constant=1.0)
    assert not hamiltonian == hamiltonian_
    assert not hamiltonian == PlumedHamiltonian(plumed_input="")

    # consider linear combination
    scaled = 0.5 * hamiltonian
    assert len(scaled) == 1
    assert scaled.get_coefficient(hamiltonian) == 0.5
    actually_scaled = EinsteinCrystal(dataset[0], force_constant=0.5)
    assert scaled.get_coefficient(actually_scaled) is None

    evaluated_scaled = scaled.evaluate(dataset[:10])
    evaluated_actually = actually_scaled.evaluate(dataset[:10])
    for i in range(1, 10):
        e = evaluated[i].result().energy
        e_scaled = evaluated_scaled[i].result().energy
        assert 0.5 * e == e_scaled
        assert 0.5 * e == evaluated_actually[i].result().energy
        assert not e == 0

    other = EinsteinCrystal(dataset[0], 4.0)
    evaluated_other = other.evaluate(dataset[:10])
    mixture = hamiltonian + other
    assert len(mixture) == 2
    assert mixture == 0.9 * other + 0.1 * other + 1.0 * hamiltonian
    _ = mixture + other
    assert mixture.get_coefficients(mixture) == (1, 1)
    assert mixture.get_coefficients(hamiltonian + actually_scaled) is None
    evaluated_mixture = mixture.evaluate(dataset[:10])
    for i in range(1, 10):
        e = evaluated[i].result().energy
        e_mixture = evaluated_mixture[i].result().energy
        e_other = evaluated_other[i].result().energy
        assert e_mixture == e + e_other

        f = evaluated[i].result().per_atom.forces
        f_mixture = evaluated_mixture[i].result().per_atom.forces
        f_other = evaluated_other[i].result().per_atom.forces
        assert np.allclose(
            f_mixture,
            f + f_other,
        )

    zero = Zero()
    evaluated_zero = zero.evaluate(evaluated)
    for i in range(evaluated_zero.length().result()):
        assert evaluated_zero[i].result().energy is None
    assert hamiltonian == hamiltonian + zero
    assert 2 * hamiltonian + zero == 2 * hamiltonian


def test_einstein_force(dataset):
    einstein = EinsteinCrystal(dataset[0], 5.0)
    reference = dataset[0].result()
    delta = 0.1
    for i in range(len(reference)):
        for j in range(3):  # x, y, z
            for sign in [+1, -1]:
                geometry = copy.deepcopy(reference)
                geometry.per_atom.positions[i, j] += sign * delta
                geometry = einstein.evaluate(Dataset([geometry]))[0].result()
                assert np.sign(geometry.per_atom.forces[i, j]) == (-1.0) * sign
                geometry.per_atom.forces[i, j] = 0.0
                assert np.allclose(geometry.per_atom.forces, 0.0)


def test_harmonic_force(dataset):
    reference_geometry = dataset[0].result()
    e = reference_geometry.energy
    harmonic = Harmonic(
        reference_geometry,
        np.eye(3 * len(reference_geometry)),
    )
    einstein = EinsteinCrystal(
        reference_geometry,
        force_constant=1,  # diagonal hessian == einstein
    )
    assert np.allclose(
        harmonic.evaluate(dataset[:10]).get("energy").result() - e,
        einstein.evaluate(dataset[:10]).get("energy").result(),
    )
    assert np.allclose(
        harmonic.evaluate(dataset[:10]).get("forces").result(),
        einstein.evaluate(dataset[:10]).get("forces").result(),
    )

    # test rudimentary __eq__
    state = dataset[3]
    h0 = Harmonic(
        state,
        np.eye(3 * len(reference_geometry)),
    )
    h1 = Harmonic(
        state,
        np.eye(3 * len(reference_geometry)),
    )
    assert h0 == h1
    h2 = Harmonic(
        dataset[3],
        np.eye(3 * len(reference_geometry)),
    )
    assert h1 != h2


def test_plumed_evaluate(context, dataset, tmp_path):
    geometry = dataset[0].result()
    atoms = Atoms(
        numbers=geometry.per_atom.numbers,
        positions=geometry.per_atom.positions,
        cell=geometry.cell,
        pbc=geometry.periodic,
    )
    center = 1
    kappa = 1
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
RESTRAINT ARG=CV AT={center} KAPPA={kappa}
""".format(
        center=center, kappa=kappa / (kJ / mol)
    )
    calculator = PlumedCalculator(plumed_input)
    # energy, forces, stress = evaluate_plumed(atoms, plumed_input)
    calculator.calculate(atoms)
    energy = calculator.results["energy"]
    forces = calculator.results["forces"]

    distance = atoms.get_distance(0, 1, mic=False)

    bias_energy = kappa / 2 * (distance - center) ** 2
    assert np.allclose(
        energy,
        bias_energy,
    )

    delta = atoms.positions[0, :] - atoms.positions[1, :]
    gradient = kappa * (distance - center) * delta / np.linalg.norm(delta)
    forces_ = np.zeros((len(atoms), 3))
    forces_[0, :] = -gradient
    forces_[1, :] = gradient
    assert np.allclose(
        forces,
        forces_,
    )

    # use external grid as bias, check that file is read
    hills = """#! FIELDS time CV sigma_CV height biasf
#! SET multivariate false
#! SET kerneltype gaussian
     1.00000     2.1     2.0     7  0
     2.00000     2.2     2.0     7  0
"""
    path_hills = tmp_path / "hills"
    with open(path_hills, "w") as f:
        f.write(hills)
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=1 SIGMA=3 HEIGHT=342 FILE={}
""".format(
        path_hills
    )
    calculator = PlumedCalculator(plumed_input, path_hills)
    for _i in range(10):
        calculator.calculate(atoms)
        energy = calculator.results["energy"]
        # energy, _, _ = evaluate_plumed(atoms, plumed_input)
    sigma = 2 * np.ones(2)
    height = np.array([7, 7]) * (kJ / mol)  # unit consistency
    center = np.array([2.1, 2.2])
    energy_ = np.sum(height * np.exp((distance - center) ** 2 / (-2 * sigma**2)))
    assert np.allclose(
        energy,
        energy_,
    )

    # check that hills file didn't change
    with open(path_hills, "r") as f:
        assert f.read() == hills


def test_plumed_hamiltonian(context, dataset, tmp_path):
    kappa = 1
    center = 100
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT={center} KAPPA={kappa}
""".format(
        center=center, kappa=kappa / (kJ / mol)
    )
    hamiltonian = PlumedHamiltonian(plumed_input)
    evaluated = hamiltonian.evaluate(dataset).geometries().result()
    for geometry in evaluated:
        volume = np.linalg.det(geometry.cell)
        assert np.allclose(
            geometry.energy,
            kappa / 2 * (volume - center) ** 2,
            atol=1e-4,
        )

    # use external grid as bias, check that file is read
    hills = """#! FIELDS time CV sigma_CV height biasf
#! SET multivariate false
#! SET kerneltype gaussian
     1.00000     2.1     2.0     7  0
     2.00000     2.2     2.0     7  0
"""
    path_hills = tmp_path / "hills"
    with open(path_hills, "w") as f:
        f.write(hills)
    data_future = copy_data_future(
        inputs=[File(path_hills)],
        outputs=[File(str(path_hills) + "_")],
    ).outputs[0]
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=1 SIGMA=3 HEIGHT=342 FILE={}
""".format(
        data_future.filepath
    )
    hamiltonian = PlumedHamiltonian(plumed_input, data_future)
    data = hamiltonian.evaluate(dataset)
    assert np.all(data.get("energy")[0].result() > 0)
    # for i in range(data.length().result()):
    #    assert data[i].result().info["energy"] > 0
    hamiltonian = PlumedHamiltonian(plumed_input, File(path_hills))
    data = hamiltonian.evaluate(dataset)
    for i in range(data.length().result()):
        assert data[i].result().energy > 0


def test_json_dump(context):
    data = {
        "a": np.ones((3, 3, 3, 2)),
        "b": [1, 2, 3],
        "c": (1, 2, 4),
        "d": "asdf",
        "e": copy_app_future(False),
    }
    data_future = dump_json(
        **data,
        outputs=[psiflow.context().new_file("bla_", ".json")],
    ).outputs[0]
    psiflow.wait()
    with open(data_future.filepath, "r") as f:
        data_ = json.loads(f.read())

    new_a = np.array(data_["a"])
    assert len(new_a.shape) == 4
    assert np.allclose(
        data["a"],
        new_a,
    )
    assert data_["e"] is False
    assert type(data_["b"]) is list
    assert type(data_["c"]) is list


def test_serialization(context, dataset, tmp_path, mace_model):
    hamiltonian = EinsteinCrystal(dataset[0], force_constant=1)
    evaluated = hamiltonian.evaluate(dataset[:3])

    data_future = hamiltonian.serialize_calculator()
    psiflow.wait()
    calculator = deserialize_calculator(data_future.filepath)
    geometry = dataset[0].result()
    atoms = Atoms(
        numbers=geometry.per_atom.numbers,
        positions=geometry.per_atom.positions,
        cell=geometry.cell,
        pbc=geometry.periodic,
    )
    atoms.calc = calculator
    for i in range(3):
        state = dataset[i].result()
        atoms.set_positions(state.per_atom.positions)
        atoms.set_cell(state.cell)
        e = atoms.get_potential_energy()
        assert np.allclose(e, evaluated[i].result().energy)

    # for plumed
    hills = """#! FIELDS time CV sigma_CV height biasf
#! SET multivariate false
#! SET kerneltype gaussian
     1.00000     2.1     2.0     7  0
     2.00000     2.2     2.0     7  0
"""
    path_hills = tmp_path / "hills"
    with open(path_hills, "w") as f:
        f.write(hills)
    data_future = copy_data_future(
        inputs=[File(path_hills)],
        outputs=[File(str(path_hills) + "_")],
    ).outputs[0]
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=1 SIGMA=3 HEIGHT=342 FILE={}
""".format(
        data_future.filepath
    )
    hamiltonian = PlumedHamiltonian(plumed_input, data_future)
    evaluated = hamiltonian.evaluate(dataset[:3])

    data_future = hamiltonian.serialize_calculator()
    psiflow.wait()
    calculator = deserialize_calculator(data_future.filepath)
    geometry = dataset[0].result()
    atoms = Atoms(
        numbers=geometry.per_atom.numbers,
        positions=geometry.per_atom.positions,
        cell=geometry.cell,
        pbc=geometry.periodic,
    )
    atoms.calc = calculator
    for i in range(3):
        state = dataset[i].result()
        atoms.set_positions(state.per_atom.positions)
        atoms.set_cell(state.cell)
        e = atoms.get_potential_energy()
        assert np.allclose(e, evaluated[i].result().energy, atol=1e-4)

    # for mace
    hamiltonian = MACEHamiltonian.from_model(mace_model)
    evaluated = hamiltonian.evaluate(dataset[:3])

    data_future = hamiltonian.serialize_calculator()
    psiflow.wait()
    calculator = deserialize_calculator(
        data_future.filepath, device="cpu", dtype="float32"
    )
    geometry = dataset[0].result()
    atoms = Atoms(
        numbers=geometry.per_atom.numbers,
        positions=geometry.per_atom.positions,
        cell=geometry.cell,
        pbc=geometry.periodic,
    )
    atoms.calc = calculator
    for i in range(3):
        state = dataset[i].result()
        atoms.set_positions(state.per_atom.positions)
        atoms.set_cell(state.cell)
        e = atoms.get_potential_energy()
        assert np.allclose(e, evaluated[i].result().energy)


def test_max_force(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=0.5)

    normal_forces = einstein.evaluate(dataset[:2]).get("forces").result()[1]
    assert np.all(np.linalg.norm(normal_forces, axis=1) < 30)

    einstein = EinsteinCrystal(dataset[0], force_constant=5000)
    large_forces = einstein.evaluate(dataset[:2]).get("forces").result()[1]
    with pytest.raises(ForceMagnitudeException):
        check_forces(large_forces, dataset[1].result(), max_force=10)


def test_subtract(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=1.0)
    h = einstein - einstein
    assert isinstance(h, MixtureHamiltonian)
    assert np.allclose(h.coefficients, 0.0)


def test_hamiltonian_serialize(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=1.0)

    kappa = 1
    center = 100
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT={center} KAPPA={kappa}
""".format(
        center=center, kappa=kappa / (kJ / mol)
    )
    plumed = PlumedHamiltonian(plumed_input)
    data = psiflow.serialize(einstein).result()
    assert "EinsteinCrystal" in data
    assert "reference_geometry" in data["EinsteinCrystal"]["_geoms"]
    einstein_ = psiflow.deserialize(data)
    assert np.allclose(
        einstein.evaluate(dataset[:10]).get("energy").result(),
        einstein_.evaluate(dataset[:10]).get("energy").result(),
    )

    mixed = 0.1 * einstein + 0.9 * plumed
    assert "hamiltonians" in mixed._serial
    assert "coefficients" in mixed._attrs
    data = psiflow.serialize(mixed).result()
    assert "MixtureHamiltonian" in data
    assert "hamiltonians" in data["MixtureHamiltonian"]["_serial"]
    mixed_ = psiflow.deserialize(data)
    for i, h in enumerate(mixed.hamiltonians):
        if isinstance(h, EinsteinCrystal):
            assert h.force_constant == mixed_.hamiltonians[i].force_constant
            assert (
                h.reference_geometry.result()
                == mixed_.hamiltonians[i].reference_geometry
            )
        else:
            assert h == mixed_.hamiltonians[i]
        assert mixed.coefficients[i] == mixed_.coefficients[i]
    assert np.allclose(
        mixed.evaluate(dataset[:10]).get("energy").result(),
        mixed_.evaluate(dataset[:10]).get("energy").result(),
    )
