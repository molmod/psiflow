import os

import numpy as np
import pytest
from ase import Atoms
from ase.units import kB
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.models import MACEModel
from psiflow.utils import copy_app_future
from psiflow.walkers import (  # noqa: F401
    BaseWalker,
    BiasedDynamicWalker,
    DynamicWalker,
    OptimizationWalker,
    PlumedBias,
    RandomWalker,
    load_walker,
)
from psiflow.walkers.dynamic import parse_openmm_output, parse_yaff_output
from psiflow.walkers.utils import get_velocities_at_temperature


def test_random_walker_multiply(dataset, tmp_path):
    amplitude_pos = 0.1
    amplitude_box = 0.0
    nwalkers = 40
    walkers = RandomWalker.multiply(
        nwalkers,
        dataset[:1],
        amplitude_pos=amplitude_pos,
        amplitude_box=amplitude_box,
    )
    for walker in walkers:
        delta = np.abs(dataset[0].result().positions - walker.state.result().positions)
        assert np.allclose(delta, 0)
        delta = np.abs(dataset[0].result().positions - walker.state0.result().positions)
        assert np.allclose(delta, 0)
    Dataset([w.propagate(None).state for w in walkers])
    for walker in walkers:
        delta = np.abs(
            walker.state0.result().positions - walker.state.result().positions
        )
        assert np.all(delta < amplitude_pos)
    walker.reset(copy_app_future(True))
    assert np.allclose(
        walker.state.result().positions,
        walker.state0.result().positions,
    )
    assert walker.counter.result() == 0


def test_walker_save_load(dataset, mace_model, tmp_path):
    walker = DynamicWalker(dataset[0], steps=10, step=1)
    path_state0 = tmp_path / "new" / "state0.xyz"
    path_state = tmp_path / "new" / "state.xyz"
    path_pars = tmp_path / "new" / "DynamicWalker.yaml"  # has name of walker class
    future, future, future = walker.save(tmp_path / "new")
    assert os.path.exists(path_state0)
    assert os.path.exists(path_state)
    assert os.path.exists(path_pars)
    walker_ = load_walker(tmp_path / "new")
    assert type(walker_) is DynamicWalker
    assert np.allclose(
        walker.state0.result().positions,
        walker_.state0.result().positions,
    )
    assert np.allclose(
        walker.state.result().positions,
        walker_.state.result().positions,
    )
    for key, value in walker.parameters.items():
        assert value == walker_.parameters[key]
    walker.propagate(model=mace_model)
    walker.save(tmp_path / "new_again")
    walker = load_walker(tmp_path / "new_again")
    assert walker.counter.result() == 10


def test_base_walker(dataset):
    walker = BaseWalker(dataset[0])
    assert isinstance(walker.state, AppFuture)
    assert isinstance(walker.state0, AppFuture)
    assert walker.state != walker.state0  # do not point to same future
    assert isinstance(walker.state0.result(), Atoms)
    assert isinstance(walker.state.result(), Atoms)

    with pytest.raises(TypeError):  # illegal kwarg
        BaseWalker(dataset[0], some_illegal_kwarg=0)


def test_random_walker(dataset):
    walker = RandomWalker(dataset[0], seed=0)

    metadata = walker.propagate()
    for key in ["counter", "state"]:
        assert key in metadata._asdict().keys()
    assert isinstance(metadata.state, AppFuture)
    assert isinstance(metadata.counter, AppFuture)
    assert isinstance(walker.is_reset(), AppFuture)
    assert not walker.is_reset().result()
    assert not walker.counter.result() == 0

    walker.reset(condition=False)
    assert not walker.is_reset().result()
    walker.reset(condition=True)
    assert walker.is_reset().result()  # should reset
    metadata = walker.propagate(model=None)  # irrelevant kwargs are ignored


def test_parse_yaff():
    stdout = """
    sampling NVT ensemble ...

  ENSEM Temperature coupling achieved through Langevin thermostat

 VERLET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 VERLET Cons.Err. = the root of the ratio of the variance on the conserved
 VERLET             quantity and the variance on the kinetic energy.
 VERLET d-rmsd    = the root-mean-square displacement of the atoms.
 VERLET g-rmsd    = the root-mean-square gradient of the energy.
 VERLET counter  Cons.Err.       Temp     d-RMSD     g-RMSD   Walltime
 VERLET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 VERLET       2    0.00000      353.7     0.0000      166.3        0.0
 VERLET       9    0.00000      355.7     0.0000      166.3        2.0
 VERLET WARNING!! You are using PLUMED as a hook for your integrator. If PLUMED
 VERLET           adds time-dependent forces (for instance when performing
 VERLET           metadynamics) there is no energy conservation. The conserved
 VERLET           quantity reported by YAFF is irrelevant in this case.
Max force exceeded: 32.15052795410156 eV/A by atom index 289
tagging sample as unsafe
    """
    counter, temperature, time = parse_yaff_output(stdout)
    assert counter == 9
    assert temperature == 354.7
    assert time == 2.0


def test_parse_openmm():
    stdout = """
PLUMED: Finished setup
PLUMED: FILE: /tmp/fileahZtE6
PLUMED: END FILE: /tmp/fileahZtE6
#"Step","Potential Energy (kJ/mole)","Temperature (K)","Box Volume (nm^3)","Elapsed Time (s)"
2,0.7217584252357483,0,0.04851977898327855,0.0004239082336425781
4,0.7216319441795349,100,0.04851977898327855,0.022877931594848633
6,0.721476137638092,200,0.04851977898327855,0.05740809440612793
8,0.7211964130401611,300,0.04851977898327855,0.0892629623413086
10,0.7210664749145508,400,0.04851977898327855,0.11220741271972656
completed all steps
current step: 10
PLUMED:                                               Cycles        Total      Average      Minimum      Max
"""
    counter, temperature, time = parse_openmm_output(stdout)
    assert counter == 10
    assert temperature == 200
    assert time == 0.11220741271972656


def test_initial_velocities():
    masses = np.random.uniform(10, 20, size=20)
    velocities = get_velocities_at_temperature(300, masses)
    actual = (velocities**2 * masses.reshape(-1, 1)).mean() / kB
    assert np.allclose(actual, 300, atol=2)


def test_dynamic_walker_plain(dataset, mace_model):
    walker = DynamicWalker(dataset[0], steps=10, step=2)
    metadata, trajectory = walker.propagate(model=mace_model, keep_trajectory=True)
    assert trajectory.length().result() == 6
    assert walker.counter.result() == 10
    assert np.allclose(
        trajectory[0].result().get_positions(),  # initial structure
        walker.state0.result().get_positions(),
    )
    assert not np.allclose(
        walker.state0.result().get_positions(),
        metadata.state.result().get_positions(),
    )

    # test timeout
    walker = DynamicWalker(dataset[0], steps=int(1e9), step=1)
    metadata, trajectory = walker.propagate(model=mace_model, keep_trajectory=True)
    assert trajectory.length().result() < int(1e9)  # timeout
    assert trajectory.length().result() > 1
    assert metadata.time.result() > 5  # ran for some time

    # test temperature reset
    walker.steps = 20
    walker.step = 5
    walker.max_excess_temperature = -10000  # always resets
    metadata = walker.propagate(model=mace_model)
    assert walker.is_reset().result()
    assert np.allclose(
        walker.state0.result().positions,
        walker.state.result().positions,
    )

    # test distance reset
    walker.steps = 20
    walker.step = 5
    walker.max_excess_temperature = 100000  # never resets
    walker.distance_threshold = 5.0  # always resets
    metadata = walker.propagate(model=mace_model)
    assert walker.is_reset().result()
    assert np.allclose(
        walker.state0.result().positions,
        walker.state.result().positions,
    )
    walker.distance_threshold = 0.5  # revert back to default

    walker.temperature = None  # NVE
    walker.force_threshold = 40
    metadata = walker.propagate(model=mace_model)
    assert not metadata.reset.result()

    walker.reset()
    walker.pressure = 0
    walker.temperature = 500
    walker.steps = 9  # takes 10 steps to attempt unit cell change
    walker.propagate(model=mace_model)
    if psiflow.context()[MACEModel][0].simulation_engine == "openmm":
        assert np.allclose(
            walker.state0.result().get_volume(),
            walker.state.result().get_volume(),
        )
    else:
        assert not np.allclose(
            walker.state0.result().get_volume(),
            walker.state.result().get_volume(),
        )
    walker.steps = 1000
    walker.max_excess_temperature = 100000  # never resets
    walker.propagate(model=mace_model)
    assert not walker.is_reset().result()
    assert not np.allclose(
        walker.state0.result().get_positions(),
        walker.state.result().get_positions(),
    )
    assert not np.allclose(
        walker.state0.result().get_volume(),
        walker.state.result().get_volume(),
    )


# def test_optimization_walker(context, dataset, mace_config):
#    training = dataset[:15]
#    validate = dataset[15:]
#    model = MACEModel(mace_config)
#    model.initialize(training)
#    model.train(training, validate)
#
#    walker = OptimizationWalker(dataset[0], optimize_cell=False, fmax=1e-1)
#    metadata, trajectory = walker.propagate(model=model, keep_trajectory=True)
#    assert trajectory.length().result() > 1
#    assert np.all(
#        np.abs(metadata.state.result().positions - dataset[0].result().positions) < 1.0
#    )
#    assert not np.all(
#        np.abs(metadata.state.result().positions - dataset[0].result().positions)
#        < 0.001
#    )  # they have to have moved
#    counter = walker.counter.result()
#    assert counter > 0
#    assert not metadata.reset.result()
#    walker.fmax = 1e-3
#    metadata = walker.propagate(model=model)
#    assert not walker.is_reset().result()
#    assert not np.all(
#        np.abs(metadata.state.result().positions - dataset[0].result().positions)
#        < 0.001
#    )  # moved again
#    assert walker.counter.result() > counter  # more steps in total


def test_biased_dynamic_walker(mace_model, dataset):
    parameters = {
        "timestep": 1,
        "steps": 10,
        "step": 1,
        "start": 0,
        "temperature": 100,
        "pressure": None,
    }

    # initial unit cell volume is around 125 A**3
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV1: MATHEVAL ARG=CV VAR=a FUNC=3*a PERIODIC=NO
restraint: RESTRAINT ARG=CV,CV1 AT=150,450 KAPPA=1,10
"""
    bias = PlumedBias(plumed_input)
    walker = BiasedDynamicWalker(dataset[0], bias=bias, **parameters)
    assert bias.components[0] == ("RESTRAINT", ("CV", "CV1"))
    metadata, trajectory = walker.propagate(model=mace_model, keep_trajectory=True)
    state = metadata.state.result()
    assert "CV" in state.info
    assert "CV1" in state.info
    state.reset()
    assert "CV" in state.info  # stays
    values = bias.evaluate(trajectory).result()
    assert np.allclose(
        3 * values[:, 0],
        values[:, 1],
    )
    assert not np.allclose(
        walker.state.result().positions,
        walker.state0.result().positions,
    )
    assert walker.counter.result() > 0
    assert not metadata.reset.result()

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
restraint: RESTRAINT ARG=CV AT=150 KAPPA=1
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
"""  # RESTART automatically added in input if not present
    bias = PlumedBias(plumed_input)
    assert ("METAD", ("CV",)) in bias.components
    assert ("RESTRAINT", ("CV",)) in bias.components
    parameters = {
        "timestep": 1,
        "steps": 10,
        "step": 1,
        "start": 0,
        "temperature": 100,
        "pressure": None,
    }
    walker = BiasedDynamicWalker(dataset[0], bias=bias, **parameters)
    walker.propagate(model=mace_model)

    with open(walker.bias.data_futures["METAD"].result(), "r") as f:
        single_length = len(f.read().split("\n"))
    assert single_length > 1
    assert walker.counter.result() > 0
    metadata = walker.propagate(model=mace_model)
    assert walker.counter.result() > 0
    assert not metadata.reset.result()
    with open(walker.bias.data_futures["METAD"].result(), "r") as f:
        double_length = len(f.read().split("\n"))
    assert double_length == 2 * single_length - 1  # twice as many gaussians
    assert double_length > 1  # have to contain some hills

    # double check MTD gives correct nonzero positive contribution
    values = walker.bias.evaluate(dataset).result()
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=2 LABEL=metad FILE=test_hills
"""  # RESTART automatically added in input if not present
    bias_mtd = PlumedBias(
        plumed_input, data={"METAD": walker.bias.data_futures["METAD"]}
    )
    values_mtd = bias_mtd.evaluate(dataset).result()
    assert np.allclose(
        values[:, 0],
        values_mtd[:, 0],
    )
    assert np.any(values_mtd[:, 1] > 0)
    bias.adjust_restraint("CV", kappa=0, center=150)  # only compute MTD contrib
    assert np.all(bias.evaluate(dataset).result()[:, 1] == 0)  # MTD bias == 0
    assert np.all(values_mtd[:, 1] >= 0)
    total = values[:, 1]
    manual = values_mtd[:, 1] + 0.5 * (values[:, 0] - 150) ** 2
    assert np.allclose(
        total,
        manual,
    )
    walker.bias = bias_mtd
    walker.reset(True)  # should also reset bias
    values = walker.bias.evaluate(dataset).result()
    assert np.allclose(
        values[:, -1],
        np.zeros(len(values[:, -1])),
    )

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
SOMEOTHER: VOLUME
restraint: RESTRAINT ARG=SOMEOTHER AT=150 KAPPA=1
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
"""  # RESTART automatically added in input if not present
    bias_ = PlumedBias(plumed_input, data={"METAD": walker.bias.data_futures["METAD"]})
    values_ = bias_.evaluate(dataset).result()
    assert np.allclose(
        values_[:, 0],
        values[:, 0],
    )


def test_walker_multiply_distribute(dataset):
    def check(walkers):
        for i, walker in enumerate(walkers):
            assert np.allclose(
                walker.state0.result().positions,
                walker.state.result().positions,
            )
            assert np.allclose(
                dataset[i].result().positions,
                walker.state.result().positions,
            )
            for j in range(i + 1, len(walkers)):
                if hasattr(walker, "bias"):  # check whether bias is copied
                    assert id(walker.bias) != id(walkers[j].bias)

    walkers = DynamicWalker.multiply(3, dataset)
    check(walkers)

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
restraint: RESTRAINT ARG=CV AT=15 KAPPA=100
"""
    bias = PlumedBias(plumed_input)
    walkers = BiasedDynamicWalker.multiply(3, dataset, bias=bias, steps=123)
    assert len(walkers) == 3
    assert type(walkers[0]) is BiasedDynamicWalker
    assert walkers[0].steps == 123
    check(walkers)

    walkers = BiasedDynamicWalker.distribute(
        nwalkers=2,
        data_start=dataset,
        bias=bias,
        variable="CV",
        min_value=0,
        max_value=50000,
    )  # should extract min and max volume state
    assert len(walkers) == 2
    volumes = [
        dataset[i].result().get_volume() for i in range(dataset.length().result())
    ]
    volume_min = min(volumes)
    volume_max = max(volumes)
    assert not volume_min == volume_max
    assert walkers[0].state.result().get_volume() == volume_min
    assert walkers[1].state.result().get_volume() == volume_max
    assert walkers[0].bias.plumed_input != walkers[1].bias.plumed_input


def test_moving_restraint_walker(dataset, mace_model):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
MOVINGRESTRAINT ARG=CV STEP0=0 AT0=150 KAPPA0=1 STEP1=1000 AT1=200 KAPPA1=1
"""
    bias = PlumedBias(plumed_input)
    values = bias.evaluate(dataset).result()
    assert np.allclose(values[:, -1], 0.0)
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=1
"""
    bias_ = PlumedBias(plumed_input)
    assert np.allclose(
        values[:, 0],
        bias_.evaluate(dataset).result()[:, 0],
    )

    walker = BiasedDynamicWalker(dataset[0], bias=bias, steps=30)
    walker.propagate(model=mace_model)
    assert not walker.is_reset().result()
