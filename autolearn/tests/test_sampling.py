import pytest
import numpy as np
from parsl.dataflow.futures import AppFuture

from ase import Atoms

from autolearn.models import NequIPModel
from autolearn.sampling import BaseWalker, Ensemble, RandomWalker, \
        DynamicWalker

from common import context, nequip_config
from test_dataset import dataset


def test_base_walker(context, dataset):
    walker = BaseWalker(context, dataset[0])
    assert isinstance(walker.start, AppFuture)
    assert isinstance(walker.start.result(), Atoms)
    assert isinstance(walker.state.result(), Atoms)
    walker.reset()
    assert walker.is_safe # reset is safe

    with pytest.raises(TypeError): # illegal kwarg
        BaseWalker(context, dataset[0], some_illegal_kwarg=0)


def test_random_walker(context, dataset):
    start = dataset[0]
    walker = RandomWalker(context, dataset[0], seed=0)
    assert np.allclose(
            start.result().get_positions(),
            walker.state.result().get_positions(),
            )
    state = walker.propagate(model=None)
    assert not np.allclose(
            start.result().get_positions(),
            walker.state.result().get_positions(),
            )
    walker_ = RandomWalker(context, dataset[0], seed=0)
    walker_.propagate(model=None)
    assert np.allclose( # same seed
            walker_.state.result().get_positions(),
            walker.state.result().get_positions(),
            )
    walker_ = RandomWalker(context, dataset[0], seed=1)
    walker_.propagate(model=None)
    assert not np.allclose( # different seed
            walker_.state.result().get_positions(),
            walker.state.result().get_positions(),
            )


def test_ensemble(context, dataset):
    walker = RandomWalker(context, dataset[0])
    ensemble = Ensemble.from_walker(walker, nwalkers=10)
    new_data = ensemble.propagate(model=None)
    for i in range(new_data.length().result() - 1):
        for j in range(i + 1, new_data.length().result()):
            assert not np.allclose(
                    new_data[i].result().get_positions(),
                    new_data[j].result().get_positions(),
                    )


def test_dynamic_walker(context, dataset, nequip_config):
    walker = DynamicWalker(context, dataset[0], steps=10, step=1)
    model  = NequIPModel(context, nequip_config)
    model.initialize(dataset[:3])
    model.deploy()
    state = walker.propagate(model)
    assert walker.tag.result() == 'safe'
    assert not np.allclose(
            walker.start.result().get_positions(),
            state.result().get_positions(),
            )
    walker.parameters.force_threshold = 0.001
    walker.parameters.steps           = 1
    walker.parameters.step            = 1
    state = walker.propagate(model)
    assert walker.tag.result() == 'unsafe' # raised ForceExceededException
