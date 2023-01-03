import pytest
import os
import numpy as np

from flower.sampling import RandomWalker, PlumedBias
from flower.ensemble import Ensemble
from flower.checks import SafetyCheck, InteratomicDistanceCheck
from flower.data import Dataset

from tests.conftest import generate_emt_cu_data


def test_ensemble_sampling(context, dataset, tmpdir):
    walker = RandomWalker(context, dataset[0])
    nwalkers = 10
    ensemble = Ensemble.from_walker(walker, nwalkers=nwalkers)

    nstates = 11
    check = SafetyCheck()
    new_data = ensemble.sample(nstates, checks=[check]) # always passes
    assert new_data.length().result() == nstates

    # no two states should be the same
    for i in range(new_data.length().result() - 1):
        for j in range(i + 1, new_data.length().result()):
            assert not np.allclose(
                    new_data[i].result().get_positions(),
                    new_data[j].result().get_positions(),
                    )

    # test save and load
    ensemble.save(tmpdir)
    ensemble_ = Ensemble.load(context, tmpdir)
    assert ensemble_.nwalkers == nwalkers
    ndirs = len([f for f in os.listdir(tmpdir) if os.path.isdir(tmpdir / f)])
    assert ndirs == nwalkers

    ensemble.walkers[3].tag_unsafe()
    ensemble.walkers[7].tag_unsafe()
    future = ensemble.reset([3, 7])
    assert not ensemble.walkers[3].is_reset().result()
    future.result() # force join_app execution
    assert ensemble.walkers[3].is_reset().result()
    assert ensemble.walkers[7].is_reset().result()
    check = SafetyCheck()
    assert check.npasses.result() == 0 # should be AppFuture
    assert check.nchecks == 0
    dataset = ensemble.as_dataset(checks=[check]) # walkers still unsafe
    assert check.nchecks == nwalkers # immediately OK because no join app
    assert dataset.length().result() == nwalkers - 2
    assert check.npasses.result() == nwalkers - 2
    dataset = ensemble.sample(nstates, model=None, checks=[check])
    assert not check.nchecks == nwalkers + nstates + 2 # because of join app!
    dataset.data_future.result() # forces join app execution
    assert check.nchecks == nwalkers + nstates + 2 # now OK
    assert check.npasses.result() == nwalkers + nstates - 2
    # unsafe walkers are reset after first pass through ensemble, so when
    # batch_size is nonzero to reach nstates samples, no unsafe walkers are
    # present.
    for walker in ensemble.walkers:
        assert walker.tag_future.result() == 'safe' # unsafe ones were reset
    check.reset()
    assert check.nchecks == 0
    assert check.npasses.result() == 0
    assert dataset.length().result() == nstates


def test_generate_distributed(context):
    data = generate_emt_cu_data(1000, 1)
    dataset = Dataset(context, atoms_list=data)
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
"""
    bias = PlumedBias(context, plumed_input)
    values = bias.evaluate(dataset, variable='CV').result()[:, 0]
    targets, step = np.linspace(
            np.min(values),
            np.max(values),
            num=10,
            retstep=True,
            )
    assert len(targets) == 10
    extracted = bias.extract_states(dataset, variable='CV', targets=targets, slack=50)
    assert extracted.length().result() == 10
    ensemble = Ensemble.from_walker(
            RandomWalker(context, dataset[0]),
            10,
            dataset=extracted,
            )

    # verify OK
    as_dataset = ensemble.as_dataset()
    values_extracted = bias.evaluate(extracted, variable='CV').result()
    values_as_dataset = bias.evaluate(as_dataset, variable='CV').result()
    assert np.allclose(values_extracted, values_as_dataset)


def test_ensemble_check(context, dataset):
    ensemble = Ensemble.from_walker(
            RandomWalker(context, dataset[0]),
            nwalkers=10,
            dataset=dataset,
            )
    ensemble.walkers[3].tag_unsafe()
    ensemble.walkers[7].tag_unsafe()
    checks = [SafetyCheck(), InteratomicDistanceCheck(threshold=0.6)]
    dataset = ensemble.sample(10, checks=checks)
    dataset.data_future.result() # necessary to force join_app execution!
    # walkers reset if unsafe
    assert ensemble.walkers[3].tag_future.result() == 'safe'
    assert ensemble.walkers[7].tag_future.result() == 'safe'
    assert len(checks[0].states.result()) == 2
