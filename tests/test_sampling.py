from dataclasses import asdict
import os
import pytest
import torch
import numpy as np
from parsl.dataflow.futures import AppFuture

from ase import Atoms

from psiflow.models import NequIPModel, MACEModel
from psiflow.sampling import BaseWalker, RandomWalker, DynamicWalker, \
        OptimizationWalker, BiasedDynamicWalker, PlumedBias, load_walker, \
        MovingRestraintDynamicWalker
from psiflow.sampling.utils import parse_yaff_output
from psiflow.data import Dataset
from psiflow.generate import generate_all


def test_random_walker_multiply(context, dataset, tmp_path):
    amplitude_pos = 0.1
    amplitude_box = 0.0
    nwalkers = 400
    walkers = RandomWalker.multiply(
            nwalkers,
            dataset[:1],
            amplitude_pos=amplitude_pos,
            amplitude_box=amplitude_box,
            )
    for i, walker in enumerate(walkers):
        delta = np.abs(dataset[0].result().positions - walker.state_future.result().positions)
        assert np.allclose(delta, 0)
        delta = np.abs(dataset[0].result().positions - walker.start_future.result().positions)
        assert np.allclose(delta, 0)
    data = generate_all(walkers, None, None, 1, 1)
    for i, walker in enumerate(walkers):
        delta = np.abs(walker.start_future.result().positions - walker.state_future.result().positions)
        assert np.all(delta < amplitude_pos)


def test_save_load(context, dataset, mace_config, tmp_path):
    walker = DynamicWalker(dataset[0], steps=10, step=1)
    path_start = tmp_path / 'start.xyz'
    path_state = tmp_path / 'state.xyz'
    path_pars  = tmp_path / 'DynamicWalker.yaml' # has name of walker class
    future, future, future = walker.save(tmp_path)
    assert os.path.exists(path_start)
    assert os.path.exists(path_state)
    assert os.path.exists(path_pars)
    walker_ = load_walker(tmp_path)
    assert type(walker_) == DynamicWalker
    assert np.allclose(
            walker.start_future.result().positions,
            walker_.start_future.result().positions,
            )
    assert np.allclose(
            walker.state_future.result().positions,
            walker_.state_future.result().positions,
            )
    for key, value in walker.parameters.items():
        assert value == walker_.parameters[key]
    model = MACEModel(mace_config)
    model.initialize(dataset[:3])
    model.deploy()
    walker.propagate(model=model)
    walker.save(tmp_path)
    walker = load_walker(tmp_path)
    assert walker.counter_future.result() == 10

def test_base_walker(context, dataset):
    walker = BaseWalker(dataset[0])
    assert isinstance(walker.state_future, AppFuture)
    assert isinstance(walker.start_future, AppFuture)
    assert walker.state_future != walker.start_future # do not point to same future
    assert isinstance(walker.start_future.result(), Atoms)
    assert isinstance(walker.state_future.result(), Atoms)

    with pytest.raises(TypeError): # illegal kwarg
        BaseWalker(dataset[0], some_illegal_kwarg=0)


def test_random_walker(context, dataset):
    walker = RandomWalker(dataset[0], seed=0)

    state = walker.propagate()
    assert isinstance(state, AppFuture)
    assert isinstance(walker.is_reset(), AppFuture)
    assert not walker.is_reset().result()
    assert not walker.counter_future.result() == 0

    walker.reset(conditional=True) # random walker is never unsafe
    assert not walker.is_reset().result()

    walker.tag_future = 'unsafe'
    walker.reset(conditional=True) # should reset
    assert walker.is_reset().result() # should reset
    assert walker.tag_future.result() == 'safe'

    state = walker.propagate(model=None) # irrelevant kwargs are ignored


def test_parse_yaff(context, dataset, tmp_path):
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
 VERLET       0    0.00000      353.7     0.0000      166.3        0.0
 VERLET       1    0.00000      353.7     0.0000      166.3        0.0
 VERLET WARNING!! You are using PLUMED as a hook for your integrator. If PLUMED
 VERLET           adds time-dependent forces (for instance when performing
 VERLET           metadynamics) there is no energy conservation. The conserved
 VERLET           quantity reported by YAFF is irrelevant in this case.
Max force exceeded: 32.15052795410156 eV/A by atom index 289
tagging sample as unsafe
    """
    tag, counter = parse_yaff_output(stdout)
    assert tag == 'unsafe'
    assert counter == 1


def test_dynamic_walker_plain(context, dataset, mace_config):
    walker = DynamicWalker(dataset[0], steps=10, step=2)
    model = MACEModel(mace_config)
    model.initialize(dataset[:3])
    model.deploy()
    state, trajectory = walker.propagate(model=model, keep_trajectory=True)
    assert trajectory.length().result() == 6
    assert walker.counter_future.result() == 10
    assert np.allclose(
            trajectory[0].result().get_positions(), # initial structure
            walker.start_future.result().get_positions(),
            )
    assert walker.tag_future.result() == 'safe'
    assert not np.allclose(
            walker.start_future.result().get_positions(),
            state.result().get_positions(),
            )

    # test timeout
    walker = DynamicWalker(dataset[0], steps=1000, step=1)
    state, trajectory = walker.propagate(model=model, keep_trajectory=True)
    assert not trajectory.length().result() < 1001 # timeout
    assert trajectory.length().result() > 1
    assert walker.counter_future.result() == trajectory.length().result() - 1
    walker.force_threshold = 1e-7 # always exceeded
    walker.steps           = 1
    walker.step            = 1
    state = walker.propagate(model=model, reset_if_unsafe=True)
    assert walker.is_reset().result()

    state = walker.propagate(model=model, reset_if_unsafe=False)
    assert walker.tag_future.result() == 'unsafe' # raised ForceExceededException

    walker.reset()
    walker.temperature = None # NVE
    state = walker.propagate(model=model)


def test_optimization_walker(context, dataset, mace_config):
    training = dataset[:15]
    validate = dataset[15:]
    model = MACEModel(mace_config)
    model.initialize(training)
    model.train(training, validate)
    model.deploy()

    walker = OptimizationWalker(dataset[0], optimize_cell=False, fmax=1e-2)
    final, trajectory = walker.propagate(model=model, keep_trajectory=True)
    assert trajectory.length().result() > 1
    assert np.all(np.abs(final.result().positions - dataset[0].result().positions) < 1.0)
    assert not np.all(np.abs(final.result().positions - dataset[0].result().positions) < 0.001) # they have to have moved
    counter = walker.counter_future.result()
    assert counter > 0
    walker.fmax = 1e-3
    final_ = walker.propagate(model=model)
    assert not np.all(np.abs(final_.result().positions - dataset[0].result().positions) < 0.001) # moved again
    assert walker.counter_future.result() > counter # more steps in total


def test_biased_dynamic_walker(context, nequip_config, dataset):
    model = NequIPModel(nequip_config)
    model.initialize(dataset)
    model.deploy()
    parameters = {
            'timestep'           : 1,
            'steps'              : 10,
            'step'               : 1,
            'start'              : 0,
            'temperature'        : 100,
            'initial_temperature': 100,
            'pressure'           : None,
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
    assert bias.components[0] == ('RESTRAINT', ('CV','CV1'))
    _, trajectory = walker.propagate(model=model, keep_trajectory=True)
    state = _.result()
    assert 'CV' in state.info
    assert 'CV1' in state.info
    state.reset()
    assert 'CV' in state.info # stays
    values = bias.evaluate(trajectory).result()
    assert np.allclose(
            3 * values[:, 0],
            values[:, 1],
            )
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
    bias = PlumedBias(plumed_input)
    assert ('METAD', ('CV',)) in bias.components
    assert ('RESTRAINT', ('CV',)) in bias.components
    parameters = {
            'timestep'           : 1,
            'steps'              : 10,
            'step'               : 1,
            'start'              : 0,
            'temperature'        : 100,
            'initial_temperature': 100,
            'pressure'           : None,
            }
    walker = BiasedDynamicWalker(dataset[0], bias=bias, **parameters)
    walker.propagate(model=model)

    with open(bias.data_futures['METAD'].result(), 'r') as f:
        single_length = len(f.read().split('\n'))
    assert walker.tag_future.result() == 'safe'
    walker.propagate(model=model)
    with open(bias.data_futures['METAD'].result(), 'r') as f:
        double_length = len(f.read().split('\n'))
    assert double_length == 2 * single_length - 1 # twice as many gaussians

    # double check MTD gives correct nonzero positive contribution
    values = walker.bias.evaluate(dataset).result()
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=2 LABEL=metad FILE=test_hills
""" # RESTART automatically added in input if not present
    bias_mtd = PlumedBias(plumed_input, data={'METAD': walker.bias.data_futures['METAD']})
    values_mtd = bias_mtd.evaluate(dataset).result()
    assert np.allclose(
            values[:, 0],
            values_mtd[:, 0],
            )
    assert np.any(values_mtd[:, 1] >  0)
    bias.adjust_restraint('CV', kappa=0, center=150) # only compute MTD contrib
    assert np.all(bias.evaluate(dataset).result()[:, 1] == 0) # MTD bias == 0
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
    bias_ = PlumedBias(plumed_input, data={'METAD': walker.bias.data_futures['METAD']})
    values_ = bias_.evaluate(dataset).result()
    assert np.allclose(
            values_[:, 0],
            values[:, 0],
            )


def test_walker_multiply_distribute(context, dataset, mace_config):
    def check(walkers):
        for i, walker in enumerate(walkers):
            assert np.allclose(
                    walker.start_future.result().positions,
                    walker.state_future.result().positions,
                    )
            assert np.allclose(
                    dataset[i].result().positions,
                    walker.state_future.result().positions,
                    )
            for j in range(i + 1, len(walkers)):
                if hasattr(walker, 'bias'): # check whether bias is copied
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
    assert type(walkers[0]) == BiasedDynamicWalker
    assert walkers[0].steps == 123
    check(walkers)

    walkers = BiasedDynamicWalker.distribute(
            nwalkers=2,
            data_start=dataset,
            bias=bias,
            variable='CV',
            min_value=0,
            max_value=50000,
            ) # should extract min and max volume state
    assert len(walkers) == 2
    volumes = [dataset[i].result().get_volume() for i in range(dataset.length().result())]
    volume_min = min(volumes)
    volume_max = max(volumes)
    assert not volume_min == volume_max
    assert walkers[0].state_future.result().get_volume() == volume_min
    assert walkers[1].state_future.result().get_volume() == volume_max
    assert walkers[0].bias.plumed_input != walkers[1].bias.plumed_input


def test_moving_restraint_walker(context, dataset, mace_config, tmp_path):
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
restraint: RESTRAINT ARG=CV AT=100.0 KAPPA=100
""" # AT=100.0 because floats are written with decimal
    bias = PlumedBias(plumed_input)
    walkers = MovingRestraintDynamicWalker.multiply(
            3,
            dataset,
            bias=bias,
            variable='CV',
            min_value=100,
            max_value=200,
            increment=50,
            num_propagations=1,
            steps=11,
            step=1,
            )
    assert walkers[0].steps == 11
    assert np.allclose(
            walkers[0].targets,
            100 + np.arange(3) * 50,
            )
    model = MACEModel(mace_config)
    model.initialize(dataset[:2])
    model.deploy()
    state, trajectory = walkers[0].propagate(model=model, keep_trajectory=True)
    assert walkers[0].counter_future.result() == 11 * 1

    walkers[0].save(tmp_path)
    walker = load_walker(tmp_path)
    assert type(walker) == MovingRestraintDynamicWalker
    assert walker.counter_future.result() == 11
    assert walker.bias.plumed_input == walkers[0].bias.plumed_input

    assert walkers[1].bias.plumed_input == walkers[2].bias.plumed_input
    state = walkers[0].propagate(model=model)
    state.result()
    assert not (walkers[0].bias.plumed_input == walkers[1].bias.plumed_input)

    state = walker.propagate(model=model) # 200
    state.result()
    state = walker.propagate(model=model) # 150
    state.result()
    state = walker.propagate(model=model) # 100
    state.result()
    assert walker.bias.plumed_input.split('\n')[-1] == walkers[1].bias.plumed_input.split('\n')[-1]

    walker.temperature = np.exp(np.log(400)) # test conversion to python native types before saving yaml
    walker.save(tmp_path)
    walker = load_walker(tmp_path)

    walker.num_propagations = 3
    walker.propagate(model=model)
    assert walker.counter_future.result() == 7 * walker.steps

    walker.reset()
    assert walker.counter_future.result() == 0

    _, trajectory = walker.propagate(model=model, keep_trajectory=True)
    assert walker.counter_future.result() == 33
    assert trajectory.length().result() == 3 * 12

    walker.force_threshold = 1e-7
    walker.propagate(model=model)
    assert walker.counter_future.result() == 0 # is reset
    assert walker.tag_future.result() == 'safe'
    assert walker.bias.plumed_input == walkers[1].bias.plumed_input
