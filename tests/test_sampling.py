from dataclasses import asdict
import os
import pytest
import torch
import numpy as np
from parsl.dataflow.futures import AppFuture

from ase import Atoms

from psiflow.models import NequIPModel, MACEModel
from psiflow.sampling import BaseWalker, RandomWalker, DynamicWalker, \
        OptimizationWalker, load_walker
from psiflow.ensemble import Ensemble
from psiflow.checks import SafetyCheck


def test_save_load(context, dataset, tmp_path):
    walker = DynamicWalker(context, dataset[0], steps=10, step=1)
    path_start = tmp_path / 'start.xyz'
    path_state = tmp_path / 'state.xyz'
    path_pars  = tmp_path / 'DynamicWalker.yaml' # has name of walker class
    future, future, future = walker.save(tmp_path)
    assert os.path.exists(path_start)
    assert os.path.exists(path_state)
    assert os.path.exists(path_pars)
    walker_ = load_walker(context, tmp_path)
    assert type(walker_) == DynamicWalker
    assert np.allclose(
            walker.start_future.result().positions,
            walker_.start_future.result().positions,
            )
    assert np.allclose(
            walker.state_future.result().positions,
            walker_.state_future.result().positions,
            )
    for key, value in asdict(walker.parameters).items():
        assert value == asdict(walker_.parameters)[key]


def test_base_walker(context, dataset):
    walker = BaseWalker(context, dataset[0])
    assert isinstance(walker.state_future, AppFuture)
    assert isinstance(walker.start_future, AppFuture)
    assert walker.state_future != walker.start_future # do not point to same future
    assert isinstance(walker.start_future.result(), Atoms)
    assert isinstance(walker.state_future.result(), Atoms)

    with pytest.raises(TypeError): # illegal kwarg
        BaseWalker(context, dataset[0], some_illegal_kwarg=0)


def test_random_walker(context, dataset):
    walker = RandomWalker(context, dataset[0], seed=0)

    state = walker.propagate()
    assert isinstance(state, AppFuture)
    assert isinstance(walker.is_reset(), AppFuture)
    assert not walker.is_reset().result()
    assert not walker.counter_future.result() == 0

    walker = RandomWalker(context, dataset[0], seed=0)
    safe_state = walker.propagate(safe_return=True)
    assert np.allclose(
            state.result().get_positions(),
            safe_state.result().get_positions(),
            )

    walker.reset(conditional=True) # random walker is never unsafe
    assert not walker.is_reset().result()

    walker.tag_future = 'unsafe'
    walker.reset(conditional=True) # should reset
    assert walker.is_reset().result() # should reset
    assert walker.tag_future.result() == 'safe'

    state = walker.propagate(model=None) # irrelevant kwargs are ignored


def test_dynamic_walker(context, dataset, mace_config):
    walker = DynamicWalker(context, dataset[0], steps=10, step=1)
    model = MACEModel(context, mace_config)
    model.initialize(dataset[:3])
    model.deploy()
    state, trajectory = walker.propagate(model=model, keep_trajectory=True)
    assert trajectory.length().result() == 11
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
    walker = DynamicWalker(context, dataset[0], steps=1000, step=1)
    state, trajectory = walker.propagate(model=model, keep_trajectory=True)
    assert not trajectory.length().result() < 1001
    assert trajectory.length().result() > 1
    assert walker.counter_future.result() == trajectory.length().result() - 1
    walker.parameters.force_threshold = 0.001
    walker.parameters.steps           = 1
    walker.parameters.step            = 1
    state = walker.propagate(model=model)
    assert walker.tag_future.result() == 'unsafe' # raised ForceExceededException


def test_optimization_walker(context, dataset, nequip_config):
    training = dataset[:15]
    validate = dataset[15:]
    model = NequIPModel(context, nequip_config)
    model.initialize(training)
    model.train(training, validate)
    model.deploy()

    walker = OptimizationWalker(context, dataset[0], optimize_cell=False, fmax=1e-2)
    final = walker.propagate(model=model)
    assert np.all(np.abs(final.result().positions - dataset[0].result().positions) < 1.0)
    assert not np.all(np.abs(final.result().positions - dataset[0].result().positions) < 0.001) # they have to have moved
    counter = walker.counter_future.result()
    assert counter > 0
    walker.parameters.fmax = 1e-3
    final_ = walker.propagate(model=model)
    assert not np.all(np.abs(final_.result().positions - dataset[0].result().positions) < 0.001) # moved again
    assert walker.counter_future.result() > counter # more steps in total
