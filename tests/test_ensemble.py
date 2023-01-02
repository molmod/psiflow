import pytest
import os
import numpy as np

from flower.sampling import RandomWalker, PlumedBias
from flower.ensemble import Ensemble
from flower.checks import SafetyCheck
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
    for i, walker in enumerate(ensemble.walkers[:int(nstates % nwalkers)]):
        assert walker.parameters.seed == i + (nstates // nwalkers + 1) * nwalkers
    for i, walker in enumerate(ensemble.walkers[int(nstates % nwalkers):]):
        assert walker.parameters.seed == i + (nstates // nwalkers) * nwalkers + nstates % nwalkers

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
    for i, walker in enumerate(ensemble_.walkers[:int(nstates % nwalkers)]):
        assert walker.parameters.seed == i + (nstates // nwalkers + 1) * nwalkers
    for i, walker in enumerate(ensemble_.walkers[int(nstates % nwalkers):]):
        assert walker.parameters.seed == i + (nstates // nwalkers) * nwalkers + nstates % nwalkers
    ndirs = len([f for f in os.listdir(tmpdir) if os.path.isdir(tmpdir / f)])
    assert ndirs == nwalkers

    ensemble.walkers[3].tag_future = 'unsafe'
    ensemble.walkers[7].tag_future = 'unsafe'
    check = SafetyCheck()
    assert check.npasses.result() == 0 # should be AppFuture
    assert check.nchecks == 0
    dataset = ensemble.as_dataset(checks=[check]) # double shouldn't matter
    assert check.nchecks == nwalkers # immediately OK because no join app
    assert dataset.length().result() == nwalkers - 2
    assert check.npasses.result() == nwalkers - 2
    dataset = ensemble.sample(nstates, model=None, checks=[check])
    assert not check.nchecks == nwalkers + nstates # because of join app!
    dataset.data_future.result() # forces join app execution
    assert check.nchecks == nwalkers + nstates # now OK
    assert check.npasses.result() == nwalkers - 2 + nstates
    check.reset()
    assert check.nchecks == 0
    assert check.npasses.result() == 0
    assert dataset.length().result() == nstates
    for walker in ensemble.walkers:
        assert walker.tag_future.result() == 'safe' # unsafe ones were reset


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
