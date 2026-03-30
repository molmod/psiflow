from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr

import psiflow
from psiflow.geometry import check_equality
from psiflow.hamiltonians import EinsteinCrystal, PlumedHamiltonian
from psiflow.models import MACE
from psiflow.sampling.optimize import (
    optimize as optimize_ipi,
    optimize_dataset as optimize_dataset_ipi,
)
from psiflow.sampling.ase import (
    optimize as optimize_ase,
    optimize_dataset as optimize_dataset_ase,
    file_ase as file_script_ase,
)
from psiflow.sampling.metadynamics import Metadynamics
from psiflow.sampling.sampling import sample, template, EnsembleTable
from psiflow.sampling.server import parse_checkpoint
from psiflow.sampling.walker import (
    Walker,
    partition,
    quench,
    randomize,
    replica_exchange,
    Ensemble,
)
from psiflow.sampling.output import Status


def test_walkers(dataset):
    mtd0 = Metadynamics("METAD: FILE=bla")  # dummy
    mtd1 = Metadynamics("METAD: FILE=bla")  # dummy
    assert not (mtd0 == mtd1)

    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
RESTRAINT ARG=CV AT=1 KAPPA=1
"""
    plumed = PlumedHamiltonian(plumed_str)
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    einstein_ = EinsteinCrystal(dataset[0], force_constant=0.2)
    walker = Walker(dataset[0], einstein, temperature=300, metadynamics=mtd0)
    assert walker.ensemble == Ensemble.NVT
    assert not walker.pimd

    walkers = [
        walker,
        Walker(dataset[0], 0.5 * einstein_, nbeads=4, metadynamics=mtd1),
        Walker(dataset[0], einstein + plumed, nbeads=4),
        Walker(dataset[0], einstein, pressure=0, temperature=300, metadynamics=mtd1),
        Walker(dataset[0], einstein_, pressure=100, temperature=600, metadynamics=mtd1),
        Walker(dataset[0], einstein, temperature=600, metadynamics=mtd1),
    ]

    # NVT
    _walkers = [walkers[0], walkers[-1]]
    hamiltonian_components, ensemble_table, plumed_list = template(_walkers)
    assert _walkers[0].ensemble == Ensemble.NVT
    assert len(hamiltonian_components) == 1
    keys_ref = ("TEMP", "EinsteinCrystal0", "METAD0", "METAD1")
    weights_ref = np.array([[300, 1, 1, 0], [600, 1, 0, 1]])
    assert ensemble_table == EnsembleTable(keys_ref, weights_ref)
    assert len(plumed_list) == 2

    # remove
    _walkers[0].metadynamics = None
    _walkers[1].metadynamics = None
    hamiltonian_components, ensemble_table, plumed_list = template(_walkers)
    assert _walkers[0].ensemble == Ensemble.NVT
    assert len(hamiltonian_components) == 1
    keys_ref = ("TEMP", "EinsteinCrystal0")
    weights_ref = np.array([[300, 1], [600, 1]])
    assert ensemble_table == EnsembleTable(keys_ref, weights_ref)
    assert len(plumed_list) == 0

    # pimd partition
    _walkers = [walkers[1], walkers[2]]
    hamiltonian_components, ensemble_table, plumed_list = template(_walkers)
    assert _walkers[0].pimd
    assert len(hamiltonian_components) == 3
    keys_ref = (
        "TEMP",
        "EinsteinCrystal0",
        "EinsteinCrystal1",
        "PlumedHamiltonian0",
        "METAD0",
    )
    weights_ref = np.array([[300, 0.0, 0.5, 0.0, 1.0], [300, 1.0, 0.0, 1.0, 0.0]])
    assert ensemble_table == EnsembleTable(keys_ref, weights_ref)

    # npt partition
    _walkers = [walkers[3], walkers[4]]
    with pytest.raises(AssertionError):  # mtd objects were equal
        _ = template(_walkers)
    _walkers[0].metadynamics = Metadynamics("METAD: FILE=bla")
    hamiltonian_components, ensemble_table, plumed_list = template(_walkers)
    assert _walkers[0].ensemble == Ensemble.NPT
    assert len(hamiltonian_components) == 2
    keys_ref = (
        "TEMP",
        "PRESSURE",
        "EinsteinCrystal0",
        "EinsteinCrystal1",
        "METAD0",
        "METAD1",
    )
    weights_ref = np.array(
        [[300, 0, 0.0, 1.0, 1.0, 0.0], [600, 100, 1.0, 0.0, 0.0, 1.0]]
    )
    assert ensemble_table == EnsembleTable(keys_ref, weights_ref)


def test_parse_checkpoint(checkpoint):
    file = psiflow.context().new_file("input_", ".xml")
    Path(file.filepath).write_text(checkpoint)
    states = parse_checkpoint(file.filepath)
    assert "time" in states[0].order
    assert np.allclose(
        states[0].cell,
        np.array([[1, 0.0, 0], [0.1, 2, 0], [0, 0, 3]]) * Bohr,
    )


def test_sample(dataset, mace_config):
    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
RESTRAINT ARG=CV AT=1 KAPPA=1
"""
    plumed = PlumedHamiltonian(plumed_str)
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)

    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
METAD ARG=CV PACE=5 SIGMA=0.05 HEIGHT=5
"""
    metadynamics = Metadynamics(plumed_str)
    walkers = [
        Walker(
            start=dataset[0],
            temperature=300,
            metadynamics=metadynamics,
            hamiltonian=0.9 * plumed + einstein,
        ),
        Walker(start=dataset[0], temperature=600, hamiltonian=einstein),
        Walker(start=dataset[0], temperature=600, hamiltonian=einstein),
        Walker(start=dataset[0], temperature=600, hamiltonian=einstein),
    ]
    outputs = sample(walkers[:2], steps=50, step=5)
    [o1] = sample(walkers[2:3], steps=20, step=2, start=10, keep_trajectory=False)
    [o2] = sample(walkers[3:], steps=20, step=2, start=10, keep_trajectory=True)

    assert o1.trajectory is None
    assert len(o1["potential{electronvolt}"].result()) == 6
    assert o2.trajectory.length().result() == 6
    assert len(o2["potential{electronvolt}"].result()) == 6

    assert outputs[0].trajectory.length().result() == 11
    pos0 = outputs[0].trajectory[-1].result().per_atom.positions
    pos1 = walkers[0].state.result().per_atom.positions
    assert np.allclose(pos0, pos1)
    assert check_equality(walkers[0].state, outputs[0].state).result()
    assert len(outputs[0]["potential{electronvolt}"].result()) == 11

    # check whether metadynamics file has correct dependency
    with open(metadynamics.external.result().filepath, "r") as f:
        content = f.read()
        nhills = len(content.split("\n"))
        assert nhills > 3

    assert len(outputs) == 2
    energies = [
        outputs[0]["potential{electronvolt}"].result(),
        outputs[1]["potential{electronvolt}"].result(),
    ]
    energies_ = [
        (0.9 * plumed + einstein).compute(outputs[0].trajectory, "energy"),
        einstein.compute(outputs[1].trajectory, "energy"),
    ]
    assert len(energies[0]) == len(energies_[0].result())
    assert np.allclose(energies[0], energies_[0].result())
    time = outputs[0]["time{picosecond}"].result()
    assert np.allclose(time, np.arange(0, 51, 5) * 5e-4)

    # check pot components
    assert np.allclose(
        outputs[1].get_energy(walkers[1].hamiltonian).result(),
        outputs[1]["potential{electronvolt}"].result(),
    )
    assert np.allclose(
        outputs[0].get_energy(walkers[0].hamiltonian).result(),
        outputs[0]["potential{electronvolt}"].result(),
    )
    manual_total = outputs[0].get_energy(plumed).result() * 0.9
    manual_total += outputs[0].get_energy(einstein).result()
    assert np.allclose(manual_total, outputs[0]["potential{electronvolt}"].result())

    observable = "ensemble_bias{electronvolt}"
    outputs = sample(walkers[:2], steps=100, step=10, observables=[observable])
    bias0 = outputs[0][observable].result()
    bias1 = outputs[1][observable].result()
    assert np.allclose(bias1, 0)
    assert np.all(bias0 > 0)

    # check that old hills are there too
    with open(metadynamics.external.result().filepath, "r") as f:
        content = f.read()
        new_nhills = len(content.split("\n"))
        assert new_nhills > 3
        assert new_nhills > nhills

    # check PIMD output
    geom_start = dataset[2]
    walker = Walker(start=geom_start, temperature=200, hamiltonian=einstein, nbeads=6)
    [output] = sample([walker], steps=10, step=5)
    for state in output.trajectory.geometries().result():
        assert len(state) == len(geom_start.result())
    assert output.trajectory.length().result() == 3
    assert output.temperature is not None
    assert np.abs(output.temperature.result() - 200 < 200)  # what?

    model = MACE(**mace_config)
    model.initialize(dataset[:3])
    hamiltonian = model.create_hamiltonian()
    walker = Walker(start=dataset[0], temperature=600, hamiltonian=hamiltonian)
    walker.state = dataset[1]
    [output] = sample(
        [walker],
        steps=10,
        observables=["kinetic_md{electronvolt}"],
        checkpoint_step=1,
        fix_com=True,  # otherwise temperature won't match
    )
    assert np.allclose(
        hamiltonian.compute(dataset, "energy").result()[0],
        output["potential{electronvolt}"].result()[0],
        atol=1e-3,
    )
    assert np.allclose(output.time.result(), 10 * 0.5 / 1000)
    temp0 = output.temperature.result()
    temp1 = output["temperature{kelvin}"].result()[-1]
    assert np.allclose(temp0, temp1)

    return


def test_ensembles(dataset, dataset_h2):
    einstein = EinsteinCrystal(dataset[0], force_constant=1e-1)
    walker_nve = Walker(dataset[0], einstein, temperature=None)
    walker_nvt = Walker(dataset[0], einstein, temperature=300)
    walker_npt = Walker(dataset[0], einstein, temperature=300, pressure=0)
    walker_nvst = Walker(dataset[0], einstein, temperature=300, volume_constrained=True)
    einstein_h2 = EinsteinCrystal(dataset_h2[3], force_constant=1e-1)
    walker_h2 = Walker(dataset_h2[0], einstein_h2)
    walkers = [walker_nve, walker_nvt, walker_npt, walker_nvst, walker_h2]

    outputs = sample(walkers, steps=20, observables=["kinetic_md{electronvolt}"])
    for output in outputs:
        assert output.status.result() == Status.DONE
        assert output.trajectory is None
    output_nve, output_nvt, output_npt, output_nvst, output_h2 = outputs

    # NVE
    energy_pot = output_nve["potential{electronvolt}"].result()
    energy_kin = output_nve["kinetic_md{electronvolt}"].result()
    energy = energy_pot + energy_kin
    assert np.allclose(energy, energy[0])

    # NVT
    volume = output_nvt["volume{angstrom3}"].result()
    assert np.allclose(volume, volume[0])

    # NPT
    volume = output_npt["volume{angstrom3}"].result()
    cell_start = walker_npt.start.result().cell
    cell_stop = output_npt.state.result().cell
    assert not np.allclose(volume, volume[0])
    assert not np.allclose(cell_start, cell_stop)

    # NVST
    volume = output_nvst["volume{angstrom3}"].result()
    cell_start = walker_nvst.start.result().cell
    cell_stop = output_nvst.state.result().cell
    assert np.allclose(volume, volume[0])
    assert not np.allclose(cell_start, cell_stop)

    # nonperiodic
    volume = output_h2["volume{angstrom3}"].result()
    assert not output_h2.state.result().periodic
    assert all(volume == 1e9)  # arbitrary large cell

    return


def test_output_status(dataset):
    """"""
    geom = dataset[0].result()
    einstein = EinsteinCrystal(geom, force_constant=0.1)
    walker = Walker(start=geom, hamiltonian=einstein)
    walker_boom = Walker(start=geom, hamiltonian=einstein, timestep=1e25, pressure=0)

    # normal & exploded
    outputs = sample([walker, walker_boom], steps=10)
    assert outputs[0].status.result() == Status.DONE
    assert outputs[1].status.result() == Status.EXPLODED

    # max force
    outputs = sample([walker], steps=100, step=1, max_force=1e-4)
    assert outputs[0].status.result() == Status.FORCE_EXCEEDED
    assert outputs[0].trajectory.length().result() == 1

    # walltime
    definition = psiflow.context().definitions["ModelEvaluation"]
    definition.max_runtime = 5  # seconds
    outputs = sample([walker], steps=10000)
    assert outputs[0].status.result() == Status.TIMEOUT
    assert outputs[0].time.result() > 0

    return


def test_reset(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    walker = Walker(start=dataset[0], temperature=300, hamiltonian=einstein)
    walker.state = dataset[1]
    assert not walker.is_reset().result()
    assert not check_equality(walker.start, walker.state).result()
    assert check_equality(walker.start, dataset[0]).result()
    assert check_equality(walker.state, dataset[1]).result()

    walker.conditional_reset(False)
    assert not walker.is_reset().result()
    walker.reset()
    assert walker.is_reset().result()
    [output] = sample([walker], steps=10)
    assert not walker.is_reset().result()


def test_quench(dataset):
    dataset = dataset[:20]
    einstein0 = EinsteinCrystal(dataset[3], force_constant=0.1)
    einstein1 = EinsteinCrystal(dataset[11], force_constant=0.1)
    walkers = Walker(start=dataset[0], hamiltonian=einstein0).multiply(10)
    walkers[2].hamiltonian = einstein1
    quench(walkers, dataset)

    assert check_equality(walkers[0].start, dataset[3]).result()
    assert check_equality(walkers[1].start, dataset[3]).result()
    assert check_equality(walkers[2].start, dataset[11]).result()
    assert check_equality(walkers[3].start, dataset[3]).result()


def test_randomize(dataset):
    walkers = Walker(dataset[0]).multiply(50)
    randomize(walkers, dataset)
    length = dataset.length().result()
    checks = []
    for i, walker in enumerate(walkers):
        assert walker.is_reset().result()
        checks.append(check_equality(walker.start, dataset[i % length]))
    checks = [bool(c.result()) for c in checks]
    assert not all(checks)


def test_rex(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    walker = Walker(dataset[0], hamiltonian=einstein, temperature=600)
    walkers = walker.multiply(2)
    replica_exchange(walkers, trial_frequency=2)
    assert walkers[0].coupling.nwalkers == len(walkers)
    assert len(partition(walkers)) == 1
    assert len(partition(walkers)[0]) == 2

    outputs = sample(walkers, steps=20, step=2)
    swaps = np.loadtxt(walkers[0].coupling.swapfile.result().filepath)
    assert outputs[0].trajectory.length().result() == 11
    assert len(swaps) > 0  # at least some successful swaps
    assert np.allclose(swaps[0, 1:], np.array([1, 0]))  # 0, 1 --> 1, 0

    walkers += Walker(dataset[0], hamiltonian=10 * einstein).multiply(2)
    with pytest.raises(AssertionError):
        replica_exchange(walkers)
    assert len(partition(walkers)) == 3
    assert partition(walkers)[0][0] == 0
    assert partition(walkers)[0][1] == 1
    assert partition(walkers)[1][0] == 2
    assert partition(walkers)[2][0] == 3


def test_walker_serialization(dataset, tmp_path):
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV PACE=2 SIGMA=0.05 HEIGHT=5
FLUSH STRIDE=1
"""
    metadynamics = Metadynamics(plumed_str)
    walkers = Walker(
        dataset[0], hamiltonian=einstein, temperature=300, metadynamics=metadynamics
    ).multiply(3)
    for i, walker in enumerate(walkers):
        walker.hamiltonian *= 1 / (1 + i)

    sample(walkers, steps=10, step=2)
    file = walkers[0].metadynamics.external.result()
    with open(file.filepath, "r") as f:
        assert len(f.read()) > 0

    data = [psiflow.serialize(obj, copy_to=tmp_path) for obj in walkers]
    new_objects = [psiflow.deserialize(d.result()) for d in data]
    psiflow.wait()
    for d in data:
        print(d.result())

    walkers_ = new_objects[:3]
    assert check_equality(walkers_[0].start, walkers[0].start).result()
    assert check_equality(walkers_[0].state, walkers[0].state).result()

    for mtd in [w.metadynamics for w in walkers_]:
        assert Path(mtd.external.filepath).exists
        with open(mtd.external.filepath, "r") as f:
            assert len(f.read()) > 0


def test_optimize_ipi(dataset):
    einstein = EinsteinCrystal(dataset[2], force_constant=10)

    # i-PI optimizer's curvature guess fails in optimum --> don't start in dataset[2]
    future, future_traj = optimize_ipi(
        dataset[0], einstein, steps=1000000, keep_trajectory=True
    )
    future_dataset = optimize_dataset_ipi(dataset[3:5], einstein, steps=1000000)

    final = future.result()
    assert np.allclose(
        final.per_atom.positions,
        dataset[2].result().per_atom.positions,
        atol=1e-4,
    )
    assert np.allclose(final.energy, 0.0)  # einstein energy >= 0
    for g in future_dataset.geometries().result():
        assert np.allclose(g.energy, 0.0)


def test_optimize_ase(dataset):
    # TODO: test applied_pressure?

    einstein = EinsteinCrystal(dataset[2], force_constant=10)
    future = optimize_ase(dataset[0], einstein, mode="fix_cell", f_max=1e-4)
    optimized = optimize_dataset_ase(
        dataset[3:5], einstein, mode="fix_cell", f_max=1e-4
    )

    final = future.result()
    assert np.allclose(
        final.per_atom.positions,
        dataset[2].result().per_atom.positions,
        atol=1e-4,
    )
    assert np.allclose(final.energy, 0.0)  # einstein energy >= 0
    for g in optimized.geometries().result():
        assert np.allclose(g.energy, 0.0)

    geom = dataset[0].result()
    plumed_input = """
    UNITS LENGTH=A ENERGY=kj/mol TIME=fs
    CV: VOLUME
    RESTRAINT ARG=CV AT=75 KAPPA=1
    """
    plumed_v = PlumedHamiltonian(plumed_input)
    final = optimize_ase(geom, plumed_v, f_max=1e-8).result()
    assert np.allclose(final.energy, 0.0) and np.allclose(final.volume, 75)

    plumed_input = """
    UNITS LENGTH=A ENERGY=kj/mol TIME=fs
    cell: CELL
    RESTRAINT ARG=cell.ax AT=3 KAPPA=1
    RESTRAINT ARG=cell.ay AT=.2 KAPPA=1
    RESTRAINT ARG=cell.az AT=.1 KAPPA=1
    """
    plumed_c = PlumedHamiltonian(plumed_input)
    future1 = optimize_ase(geom, plumed_c, mode="fix_shape", f_max=1e-8)
    future2 = optimize_ase(geom, plumed_c, mode="fix_volume", f_max=1e-8)
    future3 = optimize_ase(geom, plumed_v + plumed_c, f_max=1e-8)
    final1, final2, final3 = future1.result(), future2.result(), future3.result()

    ratio = geom.cell / final1.cell
    assert np.allclose(final1.cell[0], [3, 0, 0])
    assert np.allclose(
        ratio[ratio != 0], ratio[0, 0]
    )  # check isotropic scaling (for nonzero components)

    assert np.allclose(final2.cell[0], [3, 0.2, 0.1], atol=1e-4)
    assert np.allclose(geom.volume, final2.volume)

    assert np.allclose(final3.cell[0], [3, 0.2, 0.1], atol=1e-4)
    assert np.allclose(final3.volume, 75)

    # check 'custom' optimisation script
    final = optimize_ase(
        dataset[0], einstein, mode="fix_cell", f_max=1e-4, script=file_script_ase
    ).result()
    assert np.allclose(final.energy, 0.0)
