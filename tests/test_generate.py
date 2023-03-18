import pytest
import numpy as np
import os

from concurrent.futures import as_completed

from psiflow.data import FlowAtoms, Dataset
from psiflow.sampling import DynamicWalker, PlumedBias
from psiflow.models import MACEModel
from psiflow.checks import SafetyCheck
from psiflow.reference import EMTReference
from psiflow.generate import generate, generate_all


def test_generate_mace(context, dataset, mace_config):
    walker = DynamicWalker(dataset[0], steps=10, step=1)
    reference = EMTReference()

    model = MACEModel(mace_config)
    model.initialize(dataset[:3])
    model.deploy()
    checks = [SafetyCheck()]

    state = generate('0', walker, model, reference, 1, 1, checks=checks)
    assert checks[0].nchecks == 0   # gets executed before actual check
    state.result()                  # force check to execute
    assert checks[0].nchecks == 1   # verify check is executed
    assert checks[0].npasses == 1
    assert state.result().reference_status

    # test retry mechanism
    walker.tag_unsafe()
    state = generate('0', walker, model, reference, 1, 1, checks=checks)
    state.result() # finish everything
    assert walker.counter_future.result() == 0 # should be reset
    assert state.result().reference_status == False
    assert np.allclose(
            state.result().get_positions(),
            walker.start_future.result().get_positions(),
            )
    assert walker.counter_future.result() == 0 # walker was reset

    walker.tag_unsafe() # will retry internally
    state = generate('0', walker, model, reference, 2, 1, checks=checks)
    state.result() # finish everything
    assert walker.counter_future.result() == 10
    assert walker.tag_future.result() == 'safe'
    assert state.result().reference_status
    assert checks[0].nchecks == 4
    assert checks[0].npasses == 2

    # check whether reference energy/forces/stress are saved
    generated = Dataset([state])
    errors = Dataset.get_errors(
            generated,
            reference.evaluate(generated),
            ).result()
    assert np.allclose(errors, 0, atol=1e-3)

    # generate without reference
    state = generate('0', walker, model, None, 1, 1, checks=checks)
    assert not 'info' in state.result().info.keys()
    assert not np.allclose(
            walker.start_future.result().positions,
            state.result().positions,
            )

    # test wait_for_it
    walker.reset()
    state0 = generate('0', walker, model.copy(), None, 1, 1, checks=checks)
    state1 = generate('0', walker, model, None, 1, 1, checks=checks)
    state1.result()
    assert walker.counter_future.result() == 10 # may occasionally fail?
    walker.reset()
    state0 = generate('0', walker, model, None, 1, 1, checks=checks)
    state1 = generate('0', walker, model, None, 1, 1, state0, checks=checks)
    state1.result()
    assert walker.counter_future.result() == 20 # should never fail!

    # train model and generate afterwards
    old = model.deploy_future['float32'].filepath
    model.train(dataset[:5], dataset[5:7]) # keep_deployed == False
    assert len(model.deploy_future) == 0
    with pytest.raises(KeyError): # model not deployed
        state = generate('0', walker, model, reference, 1, 1, checks=checks)
        state.result() # force KeyError
    model.deploy()
    new = model.deploy_future['float32'].filepath
    assert old != new
    state = generate('0', walker, model, reference, 1, 1, checks=checks)
    for i, future in enumerate(as_completed([state, model.model_future])):
        if i == 0: # first, the model finishes training
            assert not isinstance(future.result(), FlowAtoms)
        else: # then, the propagation completes
            assert isinstance(future.result(), FlowAtoms)

    # train model and generate simultaneously
    old = model.deploy_future['float32'].filepath
    model.train(dataset[:5], dataset[5:7], keep_deployed=True)
    assert len(model.deploy_future) != 0
    new = model.deploy_future['float32'].filepath
    assert old == new
    state = generate('0', walker, model, reference, 1, 1, checks=checks)
    for i, future in enumerate(as_completed([state, model.model_future])):
        if i == 0: # first, the propagation finishes
            assert isinstance(future.result(), FlowAtoms)
        else: # then, the model finishes training
            assert not isinstance(future.result(), FlowAtoms)
