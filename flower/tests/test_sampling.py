import pytest
import numpy as np
from parsl.dataflow.futures import AppFuture

from ase import Atoms

from flower.models import NequIPModel
from flower.sampling import BaseWalker, Ensemble, RandomWalker, \
        DynamicWalker

from common import context, nequip_config
from test_dataset import dataset


def test_base_walker(context, dataset):
    walker = BaseWalker(context, dataset[0])
    assert isinstance(walker.state_future, AppFuture)
    assert isinstance(walker.start_future, AppFuture)
    assert walker.state_future != walker.start_future # do not point to same future
    assert isinstance(walker.start_future.result(), Atoms)
    assert isinstance(walker.state_future.result(), Atoms)
    #walker.reset_if_unsafe()
    #assert 

    with pytest.raises(TypeError): # illegal kwarg
        BaseWalker(context, dataset[0], some_illegal_kwarg=0)


def test_random_walker(context, dataset):
    walker = RandomWalker(context, dataset[0], seed=0)

    state = walker.propagate()
    assert isinstance(state, AppFuture)
    assert isinstance(walker.is_reset(), AppFuture)
    assert not walker.is_reset().result()

    walker = RandomWalker(context, dataset[0], seed=0)
    safe_state = walker.propagate(safe_return=True)
    assert np.allclose(
            state.result().get_positions(),
            safe_state.result().get_positions(),
            )

    walker.reset_if_unsafe() # random walker is never unsafe
    assert not walker.is_reset().result()

    walker.tag_future = 'unsafe'
    walker.reset_if_unsafe() # should reset
    assert walker.is_reset().result() # should reset

    state = walker.propagate(model='dummy') # irrelevant kwargs are ignored


def test_ensemble(context, dataset):
    walker = RandomWalker(context, dataset[0])
    ensemble = Ensemble.from_walker(walker, nwalkers=10)
    new_data = ensemble.propagate()
    assert ensemble.nwalkers == new_data.length().result()
    for i in range(new_data.length().result() - 1):
        for j in range(i + 1, new_data.length().result()):
            assert not np.allclose(
                    new_data[i].result().get_positions(),
                    new_data[j].result().get_positions(),
                    )


def test_dynamic_walker(context, dataset, nequip_config):
    walker = DynamicWalker(context, dataset[0], steps=10, step=1)
    model = NequIPModel(context, nequip_config, dataset[:3])
    model.deploy()
    state = walker.propagate(model=model)
    #assert walker.tag.result() == 'safe'
    assert not np.allclose(
            walker.start_future.result().get_positions(),
            state.result().get_positions(),
            )
    walker.parameters.force_threshold = 0.001
    walker.parameters.steps           = 1
    walker.parameters.step            = 1
    state = walker.propagate(model=model)
    assert walker.tag_future.result() == 'unsafe' # raised ForceExceededException
